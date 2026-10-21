import numpy as np
import mesa
import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import math
import sys
from typing import Optional, List, Dict, Any

# ─────────────────────────── CONSTANTS & CONFIG ────────────────────────



# ═════════════════════ PATIENT PAIN DYNAMICS ═════════════════════

# Pain inertia. Each step: pain = ALPHA * previous pain + (1 - ALPHA) * profile target + U(-0.3, 0.3).
# "Previous pain" is the post-treatment value, so ALPHA also sets how long a dose's relief persists.
# Higher = smoother pain that follows the profile more slowly.
ALPHA: float = 0.2

# ═════════════════════ DOCTOR DECISION RULE ═════════════════════

# SCORE doctor only. effective_score = pain_component - addiction_score * this weight.
# Higher = more cautious prescribing for patients with high addiction risk.
ADDICTION_PENALTY_WEIGHT: float = 0.3

# Tier cutoffs. Dual use:
#  (a) SCORE doctor: effective score >= HIGH -> HIGH tier, >= MED -> MEDIUM tier, else LOW.
#  (b) raw_tier(): the same cutoffs applied to raw pain. Used by the RAW_PAIN doctor
#      and to count "downgrades" for the other doctors.
# So changing these moves the baseline policy too, not just the SCORE policy.
HIGH_SCORE_THRESHOLD: float = 6.0
MED_SCORE_THRESHOLD: float = 3.0

# Length (in steps) of the rolling window for addiction AUC, which drives the addiction
# zone below. Shorter = reacts quickly to recent addiction; longer = slower, smoother memory.
# (Fixes lifetime time-step dilution)
AUC_MEM_WINDOW: int = 10

# Per-dose effect of each tier.
#   reduction:        pain points removed per dose, before zone penalty and tolerance attenuation.
#   addiction_weight: addiction score added per dose.
# LOW is "free": no addiction, tolerance, craving, zone, or tolerance-attenuation effects.
TIER_EFFECTS = {
    "LOW": {"reduction": 1.5, "addiction_weight": 0.0},
    "MEDIUM": {"reduction": 3.0, "addiction_weight": 1.0},
    "HIGH": {"reduction": 5.0, "addiction_weight": 3.0},
}

# Exogenous pain trajectory shapes; one is drawn uniformly per patient. Categorical.
# The shape parameters (sine amplitude 2.0 and period 5, decay rate 0.1, slope 0.1,
# random-walk step 0.4) are hardcoded inside PainProfile, not constants here.
PAIN_PROFILES = ["SINE", "EXP_DECAY", "LINEAR_INC", "RANDOM_WALK"]

# How the addiction score recovers each step; assigned per patient if not specified.
#   CONSTANT: subtract a fixed amount. LINEAR: subtract an amount that grows with t.
#   EXP: multiplicative decay.
# The rates (0.5, 0.02, 0.3) are hardcoded defaults in AddictionDynamic.
ADDICTION_MODES = ["CONSTANT", "LINEAR", "EXP"]

# ═════════════════════ ADDICTION ZONES (feedback loop) ═════════════════════

# Zone boundaries on mean addiction over the last AUC_MEM_WINDOW steps:
#   mean < LOW -> zone 0;  LOW <= mean < HIGH -> zone 1;  mean >= HIGH -> zone 2.
ADDICTION_AUC_LOW_THRESHOLD: float = 2.0
ADDICTION_AUC_HIGH_THRESHOLD: float = 5.0

# Consequences of each zone, applied to MEDIUM/HIGH doses only.
#   reduction_penalty: subtracted from the dose's pain relief (reduced efficacy under heavy addiction burden).
#   addiction_increase: added on top of the tier's addiction_weight (reinforcing loop).
# Zone 0 is the neutral baseline; keep it at zero.
ADDICTION_AUC_EFFECTS = {
    0: {"reduction_penalty": 0.0, "addiction_increase": 0.0},
    1: {"reduction_penalty": 0.5, "addiction_increase": 0.1},
    2: {"reduction_penalty": 1.0, "addiction_increase": 0.2},
}

# ═════════════════════ PAIN "OVERRIDE" (breakthrough pain) ═════════════════════
# Above PAIN_CRITICAL_THRESHOLD, the pain term in the treatment score explodes so that no realistic
# addiction-risk or tolerance penalty can pull effective_pain_score back below HIGH_SCORE_THRESHOLD.
# Models acute/breakthrough pain clinically overriding addiction-risk caution.
# Active only when PAIN_CONVEX_SOFT is False. Also feeds into compute_functionality, so it
# inflates the pain term of impairment, not just the doctor's decision.
#   pain_component = pain + COEF * (pain - CRITICAL) ** POWER   for pain > CRITICAL
PAIN_CRITICAL_THRESHOLD: float = 8.0   # pain level where the override kicks in
PAIN_EXPLOSION_COEF: float = 5.0       # scale of the override
PAIN_EXPLOSION_POWER: float = 3.0      # steepness above the threshold

# ═════════════════════ ADDICTION NOISE ═════════════════════

# Std dev of mean-zero Gaussian noise added to addiction score each step (floored at 0).
# Makes addiction risk volatile rather than moving purely on rails.
ADDICTION_RISK_NOISE_STD: float = 0.15

# ═════════════════════ TOLERANCE ═════════════════════
# Tolerance (0-10) builds with MEDIUM/HIGH doses and erodes pain relief (Emax-style).

TOLERANCE_GAIN_MEDIUM: float = 0.4   # tolerance added per MEDIUM dose
TOLERANCE_GAIN_HIGH: float = 0.8     # tolerance added per HIGH dose
TOLERANCE_DECAY_RATE: float = 0.05   # fraction of tolerance lost per step (natural washout)

# Dual role:
#  (1) Inflection point (Km) of efficacy loss: efficacy = 1 / (1 + (tolerance / TOLERANCE_50) ** 2),
#      so efficacy is 50% at tolerance = TOLERANCE_50 (Hill exponent fixed at 2).
#  (2) SCORE doctor: above this tolerance, effective score drops by 0.2 * tolerance (0.2 hardcoded).
TOLERANCE_50: float = 5.0

# ═════════════════════ CRAVING ═════════════════════
# Craving (0-10) rises with MEDIUM/HIGH doses and falls with time and pain.
# It does not feed back into addiction or the SCORE doctor; it affects only functionality.

CRAVING_GAIN_PER_DOSE: float = 0.6     # base craving added per MEDIUM/HIGH dose
CRAVING_TOLERANCE_SCALE: float = 0.1   # extra craving per unit of (post-dose) tolerance
CRAVING_DECAY_RATE: float = 0.08       # fraction of craving lost per step
CRAVING_PAIN_SUPPRESSION: float = 0.05 # craving subtracted per step = this * current pain
CRAVING_BOUNDS = (0.0, 10.0)           # clamp on craving

# ═════════════════════ FUNCTIONALITY ═════════════════════
# impairment = weighted sum of the five terms below; functionality = 10 * (1 - sigmoid-squash(impairment)).
# Weights sum to 1.0. Pain, addiction, tolerance, and craving are each scaled to roughly 0-1 first.

FUNC_WEIGHT_PAIN: float = 0.35            # impairment from pain (uses the exploded pain term)
FUNC_WEIGHT_ADDICTION: float = 0.25       # impairment from addiction score
FUNC_WEIGHT_TOLERANCE: float = 0.15       # impairment from tolerance
FUNC_WEIGHT_CRAVING: float = 0.15         # impairment from craving
FUNC_WEIGHT_TREATMENT_TIER: float = 0.10  # impairment from the dose itself (sedation/side effects)

# Side-effect strength of the dose taken this step, normalized by 3 in the impairment formula.
# None = untreated. Independent of the per-tier pain reductions in TIER_EFFECTS.
TIER_STRENGTH_VALUES = {"HIGH": 3.0, "MEDIUM": 1.0, "LOW": 0.3, None: 0.0}

# Smooth alternative to the override above; used only if PAIN_CONVEX_SOFT is True:
#   pain_component = pain + COEF * (pain / 10) ** POWER
PAIN_CONVEX_COEF: float = 12.0   # strength of the convex term
PAIN_CONVEX_POWER: float = 4.0   # curvature; higher = more weight concentrated at the top
# Selects which compute_pain_component is defined AT IMPORT TIME: False = threshold override, True = smooth convex.
PAIN_CONVEX_SOFT: bool = False

# Impairment -> functionality squash.
FUNC_SIGMOID_MID: float = 0.5   # impairment at which functionality is ~halfway down
FUNC_SIGMOID_K: float = 4.0     # steepness of the transition

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

# ═════════════════════ OUTCOME DEFINITIONS (not model dynamics) ═════════════════════

# Post-treatment pain above this counts toward time_severe_pain. Coincidentally equal to
# PAIN_CRITICAL_THRESHOLD but independent of it; changing one will not change the other.
SEVERE_PAIN_THRESHOLD: float = 8.0

# Per-trajectory outcomes reported by report_summaries and used as Sobol targets.
SUMMARY_METRICS = ["pain_auc", "time_severe_pain", "addiction_auc", "high_share", "max_high_streak"]

def _auc(series) -> float:
    s = np.asarray(series, dtype=float)
    if len(s) < 2:
        return 0.0
    return float(np.trapezoid(s) if hasattr(np, "trapezoid") else np.trapz(s))


def _max_streak(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def trajectory_summary(pdata) -> Dict[str, float]:
    """Policy-level outcomes for one patient trajectory (pdata = that patient's collected frame)."""
    pain = pdata["Pain"].to_numpy(dtype=float)               # post-treatment pain
    addiction = pdata["AddictionScore"].to_numpy(dtype=float)
    tiers = list(pdata["Tier"])                              # None/NaN = untreated this step
    high = [t == "HIGH" for t in tiers]
    n_treated = sum(t in TIER_EFFECTS for t in tiers)
    return {
        "pain_auc": _auc(pain),
        "time_severe_pain": int(np.sum(pain > SEVERE_PAIN_THRESHOLD)),
        "addiction_auc": _auc(addiction),
        "high_share": sum(high) / n_treated if n_treated else np.nan,
        "max_high_streak": _max_streak(high),
    }

# ─────────────────────────── COMPONENT DYNAMICS ────────────────────────

class PainProfile:
    def __init__(self, profile_type: str, base_pain: float, rng: random.Random):
        self.profile_type = profile_type.upper()
        self.base_pain = base_pain
        self._rw_current = base_pain
        self.rng = rng

    def target(self, t: int) -> float:
        if self.profile_type == "SINE":
            return max(0.0, min(10.0, self.base_pain + 2.0 * math.sin(t / 5.0)))
        elif self.profile_type == "EXP_DECAY":
            return max(0.0, min(10.0, 1.0 + (self.base_pain - 1.0) * math.exp(-0.1 * t)))
        elif self.profile_type == "LINEAR_INC":
            return max(0.0, min(10.0, self.base_pain + 0.1 * t))
        elif self.profile_type == "RANDOM_WALK":
            self._rw_current += self.rng.uniform(-0.4, 0.4)
            self._rw_current = max(0.0, min(10.0, self._rw_current))
            return self._rw_current
        else:
            raise ValueError(f"Unknown profile type: {self.profile_type}")


class AddictionDynamic:
    def __init__(self, mode: str = "CONSTANT", base_rate: float = 0.5, decay_rate: float = 0.02, lambda_: float = 0.3):
        self.mode = mode.upper()
        self.base_rate = base_rate
        self.decay_rate = decay_rate
        self.lambda_ = lambda_

    def decay(self, current_score: float, t: int) -> float:
        if self.mode == "CONSTANT":
            return max(0.0, current_score - self.base_rate)
        elif self.mode == "LINEAR":
            return max(0.0, current_score - (self.base_rate * (1.0 + self.decay_rate * t)))
        elif self.mode == "EXP":
            return current_score * math.exp(-self.lambda_)
        return current_score


class ToleranceDynamic:
    def decay_step(self, current_tolerance: float) -> float:
        return max(0.0, current_tolerance * (1.0 - TOLERANCE_DECAY_RATE))

    def gain_from_dose(self, tier_name: str) -> float:
        if tier_name == "MEDIUM": return TOLERANCE_GAIN_MEDIUM
        if tier_name == "HIGH":   return TOLERANCE_GAIN_HIGH
        return 0.0

    def apply_to_reduction(self, base_reduction: float, tolerance: float) -> float:
        if tolerance <= 0: return base_reduction
        efficacy_multiplier = 1.0 / (1.0 + (tolerance / TOLERANCE_50) ** 2)
        return base_reduction * efficacy_multiplier


class CravingDynamic:
    def decay_step(self, current_craving: float, current_pain: float) -> float:
        craving = current_craving * (1.0 - CRAVING_DECAY_RATE)
        craving -= CRAVING_PAIN_SUPPRESSION * current_pain
        return max(CRAVING_BOUNDS[0], min(CRAVING_BOUNDS[1], craving))

    def gain_from_dose(self, tier_name: str, tolerance: float) -> float:
        if tier_name in ("MEDIUM", "HIGH"):
            return CRAVING_GAIN_PER_DOSE + tolerance * CRAVING_TOLERANCE_SCALE
        return 0.0


if not PAIN_CONVEX_SOFT:
    def compute_pain_component(pain: float) -> float:
        """
        Transforms raw pain into the pain term used for the doctor's treatment
        score. Below PAIN_CRITICAL_THRESHOLD this is just identity (pain itself).
        Above it, the term grows as a power-law function of the excess pain, so
        breakthrough/acute pain (>8) produces a treatment score large enough that
        subtracting the addiction-risk penalty and tolerance penalty cannot drop
        it below HIGH_SCORE_THRESHOLD -> the doctor treats regardless of addiction
        risk. This is intentionally steep; tune PAIN_EXPLOSION_COEF/POWER to taste.
        """
        if pain <= PAIN_CRITICAL_THRESHOLD:
            return pain
        excess = pain - PAIN_CRITICAL_THRESHOLD
        return pain + PAIN_EXPLOSION_COEF * (excess ** PAIN_EXPLOSION_POWER)

elif PAIN_CONVEX_SOFT:
    def compute_pain_component(pain: float) -> float:

        """Smooth convex transform of raw pain for the doctor's treatment score.
        f(p) = p + A * (p/10)^k   -> monotone, convex, no kink.
        Low pain stays ~linear; high pain gets a disproportionately large score,
        so the addiction-risk and tolerance penalties are increasingly outweighed. """

        return pain + PAIN_CONVEX_COEF * (pain / 10.0) ** PAIN_CONVEX_POWER


def compute_functionality(pain: float, addiction: float, tolerance: float, craving: float,
                          current_tier: Optional[str]) -> float:
    tier_strength = TIER_STRENGTH_VALUES.get(current_tier, 0.0)

    if not PAIN_CONVEX_SOFT:
        pain = compute_pain_component(pain)
    else:
        pain = compute_pain_component(pain)

    impairment = ((
            FUNC_WEIGHT_PAIN * pain / 10.0) + FUNC_WEIGHT_ADDICTION * min(1.0, addiction / 10.0)
            + FUNC_WEIGHT_TOLERANCE * min(1.0, tolerance / 10.0) +FUNC_WEIGHT_CRAVING * (craving / 10.0)
            +FUNC_WEIGHT_TREATMENT_TIER * (tier_strength / 3.0) )

    # Rescale so impairment 0 -> functionality 10 exactly; the upper end is asymptotic to 0.
    lo = sigmoid(-FUNC_SIGMOID_K * FUNC_SIGMOID_MID)
    g = (sigmoid(FUNC_SIGMOID_K * (impairment - FUNC_SIGMOID_MID)) - lo) / (1.0 - lo)
    return 10.0 * (1.0 - g)


# ─────────────────────────── AGENTS ────────────────────────

class PatientAgent(mesa.Agent):
    def __init__(self, model, base_pain: float = 5.0, addiction_decay_mode: str = "CONSTANT", rng = None):
        super().__init__(model)
        self.local_rng = rng if rng is not None else random.Random()

        self.base_pain = base_pain
        self.pain = base_pain
        self.pain_pretreatment = base_pain  # raw severity before any treatment this step

        self.profile_type = self.local_rng.choice(PAIN_PROFILES)
        self.profile = PainProfile(self.profile_type, base_pain, self.local_rng)

        self.addiction_dynamic = AddictionDynamic(mode=addiction_decay_mode)
        self.tolerance_dynamic = ToleranceDynamic()
        self.craving_dynamic = CravingDynamic()

        self.addiction_score = 0.0
        self.tolerance = 0.0
        self.craving = 0.0
        self.functionality = 10.0

        self.treated_this_step = False
        self.current_tier = None
        self.medium_treatment_count = 0
        self.high_treatment_count = 0
        self.downgrades_received = 0
        self.steps_since_last_dose = 0

        self.pain_history: List[float] = [base_pain]
        self.addiction_history: List[float] = [0.0]
        self.profile_targets: List[float] = []

    @property
    def rolling_pain_auc(self) -> float:
        history = self.pain_history[-AUC_MEM_WINDOW:]
        if len(history) < 2: return 0.0
        if hasattr(np, 'trapezoid'):
            return float(np.trapezoid(history))
        return float(np.trapz(history))

    @property
    def rolling_addiction_auc(self) -> float:
        history = self.addiction_history[-AUC_MEM_WINDOW:]
        if len(history) < 2: return 0.0
        if hasattr(np, 'trapezoid'):
            return float(np.trapezoid(history))
        return float(np.trapz(history))

    def pre_treatment_update(self):
        t = self.model.current_step
        target = self.profile.target(t)
        self.profile_targets.append(target)

        noise = self.local_rng.uniform(-0.3, 0.3)
        self.pain = max(0.0, min(10.0, ALPHA * self.pain + (1.0 - ALPHA) * target + noise))
        self.pain_pretreatment = self.pain  # snapshot raw severity before the doctor sees it

        self.steps_since_last_dose += 1
        self.addiction_score = self.addiction_dynamic.decay(self.addiction_score, t)
        # Volatility: addiction risk doesn't move purely on rails - add mean-0
        # Gaussian noise each step, floored at 0 since risk can't go negative.
        self.addiction_score = max(0.0, self.addiction_score + self.local_rng.gauss(0.0, ADDICTION_RISK_NOISE_STD))
        self.tolerance = self.tolerance_dynamic.decay_step(self.tolerance)
        self.craving = self.craving_dynamic.decay_step(self.craving, self.pain)

        self.treated_this_step = False
        self.current_tier = None

    def _treatment_outcome(self, tier_name: Optional[str]) -> Dict[str, float]:
        """Pure function: post-dose state for a tier. Does not mutate the patient."""
        if tier_name is None:
            return {"pain": self.pain, "addiction": self.addiction_score,
                    "tolerance": self.tolerance, "craving": self.craving}

        effects = TIER_EFFECTS[tier_name]
        reduction = effects["reduction"]
        addiction_weight = effects["addiction_weight"]

        avg_addiction = self.rolling_addiction_auc / max(1, min(self.model.current_step, AUC_MEM_WINDOW))
        zone = 0
        if avg_addiction >= ADDICTION_AUC_HIGH_THRESHOLD:
            zone = 2
        elif avg_addiction >= ADDICTION_AUC_LOW_THRESHOLD:
            zone = 1

        zone_fx = ADDICTION_AUC_EFFECTS[zone]
        if tier_name in ("MEDIUM", "HIGH"):
            reduction = max(0.0, reduction - zone_fx["reduction_penalty"])
            addiction_weight += zone_fx["addiction_increase"]
            effective_reduction = self.tolerance_dynamic.apply_to_reduction(reduction, self.tolerance)
        else:
            effective_reduction = reduction

        new_tolerance = min(10.0, self.tolerance + self.tolerance_dynamic.gain_from_dose(tier_name))
        # Craving gain uses the post-dose tolerance (same order as the original code)
        new_craving = min(CRAVING_BOUNDS[1],
                          self.craving + self.craving_dynamic.gain_from_dose(tier_name, new_tolerance))
        return {
            "pain": max(0.0, self.pain - effective_reduction),
            "addiction": self.addiction_score + addiction_weight,
            "tolerance": new_tolerance,
            "craving": new_craving,
        }

    def predicted_functionality(self, tier_name: Optional[str]) -> float:
        out = self._treatment_outcome(tier_name)
        return compute_functionality(out["pain"], out["addiction"], out["tolerance"],
                                     out["craving"], tier_name)

    def receive_treatment(self, tier_name: str):
        out = self._treatment_outcome(tier_name)
        self.pain = out["pain"]
        self.addiction_score = out["addiction"]
        self.tolerance = out["tolerance"]
        self.craving = out["craving"]
        self.treated_this_step = True
        self.current_tier = tier_name

        if tier_name == "MEDIUM":
            self.medium_treatment_count += 1
        elif tier_name == "HIGH":
            self.high_treatment_count += 1
        if tier_name in ("MEDIUM", "HIGH"):
            self.steps_since_last_dose = 0

    def post_treatment_cleanup(self):
        self.functionality = compute_functionality(
            self.pain, self.addiction_score, self.tolerance, self.craving, self.current_tier
        )
        self.pain_history.append(self.pain)
        self.addiction_history.append(self.addiction_score)


class DoctorAgent(mesa.Agent):
    def __init__(self, model, pain_threshold: float = 3.0, max_treatments_per_step: int = 2):
        super().__init__(model)
        self.pain_threshold = pain_threshold
        self.max_treatments_per_step = max_treatments_per_step
        self.treatments_given = 0
        self.downgrades_issued = 0

    def step(self):
        treatments_this_step = 0
        patients = list(self.model.agents_by_type[PatientAgent])
        patients.sort(key=lambda p: p.pain, reverse=True)

        for patient in patients:
            if treatments_this_step >= self.max_treatments_per_step:
                break

            if patient.pain < self.pain_threshold:
                continue

            pain_component = compute_pain_component(patient.pain)
            effective_pain_score = pain_component - (patient.addiction_score * ADDICTION_PENALTY_WEIGHT)

            # Fixed: Tied to inflection constants dynamically instead of hardcoded 6.0
            if patient.tolerance > TOLERANCE_50:
                effective_pain_score -= (patient.tolerance * 0.2)

            if effective_pain_score >= HIGH_SCORE_THRESHOLD:
                chosen_tier = "HIGH"
            elif effective_pain_score >= MED_SCORE_THRESHOLD:
                chosen_tier = "MEDIUM"
            else:
                chosen_tier = "LOW"

            if patient.pain >= HIGH_SCORE_THRESHOLD:
                raw_tier = "HIGH"
            elif patient.pain >= MED_SCORE_THRESHOLD:
                raw_tier = "MEDIUM"
            else:
                raw_tier = "LOW"

            tier_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            if tier_order[chosen_tier] < tier_order[raw_tier]:
                self.downgrades_issued += 1
                patient.downgrades_received += 1

            patient.receive_treatment(chosen_tier)
            treatments_this_step += 1
            self.treatments_given += 1


def raw_tier(pain: float) -> str:
    if pain >= HIGH_SCORE_THRESHOLD: return "HIGH"
    if pain >= MED_SCORE_THRESHOLD:  return "MEDIUM"
    return "LOW"


class FunctionalityDoctorAgent(DoctorAgent):
    """Picks, per patient, the action (none/LOW/MEDIUM/HIGH) that maximizes
    predicted post-treatment functionality. Under a per-step cap, treats the
    patients with the largest functionality gain first."""

    def __init__(self, model, pain_threshold: float = 3.0, max_treatments_per_step: int = 2,
                 min_gain: float = 0.0):
        super().__init__(model, pain_threshold, max_treatments_per_step)
        self.min_gain = min_gain  # minimum functionality gain required to treat

    def step(self):
        plans = []
        for patient in self.model.agents_by_type[PatientAgent]:
            if patient.pain < self.pain_threshold:
                continue
            baseline = patient.predicted_functionality(None)
            best_tier, best_f = None, baseline
            for tier in ("LOW", "MEDIUM", "HIGH"):
                f = patient.predicted_functionality(tier)
                if f > best_f:
                    best_tier, best_f = tier, f
            gain = best_f - baseline
            if best_tier is not None and gain > self.min_gain:
                plans.append((gain, patient, best_tier))

        plans.sort(key=lambda x: x[0], reverse=True)
        tier_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        for gain, patient, tier in plans[: self.max_treatments_per_step]:
            if tier_order[tier] < tier_order[raw_tier(patient.pain)]:
                self.downgrades_issued += 1
                patient.downgrades_received += 1
            patient.receive_treatment(tier)
            self.treatments_given += 1

class RawPainDoctorAgent(DoctorAgent):
    """Baseline: tier set by raw pain alone, ignoring addiction and tolerance.
    Same ordering and per-step cap as DoctorAgent, so only the tier rule differs."""

    def step(self):
        treatments_this_step = 0
        patients = sorted(self.model.agents_by_type[PatientAgent],
                          key=lambda p: p.pain_pretreatment, reverse=True)
        for patient in patients:
            if treatments_this_step >= self.max_treatments_per_step:
                break
            if patient.pain_pretreatment < self.pain_threshold:
                continue
            patient.receive_treatment(raw_tier(patient.pain_pretreatment))
            treatments_this_step += 1
            self.treatments_given += 1



# ─────────────────────────── MODEL ENVIRONMENT ────────────────────────

class DoctorPatientModel(mesa.Model):
    def __init__(self, n_patients=1, pain_threshold=3.0, max_treatments_per_step=2,
                 addiction_decay_mode=None, doctor_type="SCORE", seed=None):
        super().__init__()
        self.current_step = 0
        cohort_rng = random.Random(seed)

        for _ in range(n_patients):
            base_pain = cohort_rng.uniform(3.0, 8.0)
            mode = cohort_rng.choice(ADDICTION_MODES) if addiction_decay_mode is None else addiction_decay_mode
            patient_rng = random.Random(cohort_rng.getrandbits(64))  # independent stream per patient
            PatientAgent(model=self, base_pain=base_pain,
                         addiction_decay_mode=mode, rng=patient_rng)



        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Pain": lambda a: getattr(a, 'pain', None) if isinstance(a, PatientAgent) else None,
                "PainPreTreatment": lambda a: getattr(a, 'pain_pretreatment', None) if isinstance(a, PatientAgent) else None,
                "Tier": lambda a: getattr(a, 'current_tier', None) if isinstance(a, PatientAgent) else None,
                "AddictionScore": lambda a: getattr(a, 'addiction_score', None) if isinstance(a,
                                                                                              PatientAgent) else None,
                "Tolerance": lambda a: getattr(a, 'tolerance', None) if isinstance(a, PatientAgent) else None,
                "Craving": lambda a: getattr(a, 'craving', None) if isinstance(a, PatientAgent) else None,
                "Functionality": lambda a: getattr(a, 'functionality', None) if isinstance(a, PatientAgent) else None,
                "ProfileType": lambda a: getattr(a, 'profile_type', None) if isinstance(a, PatientAgent) else None,
            }
        )

        self.doctor_cls = {"SCORE": DoctorAgent,
                          "FUNCTIONALITY": FunctionalityDoctorAgent,
                          "RAW_PAIN": RawPainDoctorAgent}[doctor_type.upper()]
        self.doctor_cls(model=self, pain_threshold=pain_threshold,
                        max_treatments_per_step=max_treatments_per_step)

    def step(self):
        patients = self.agents_by_type[PatientAgent]
        doctors = self.agents_by_type[self.doctor_cls]

        patients.do("pre_treatment_update")
        doctors.do("step")
        patients.do("post_treatment_cleanup")

        self.datacollector.collect(self)
        self.current_step += 1


# ─────────────────────────── OPTIMIZED PROCESSING & PLOTS ────────────────

def run_batch_by_profile(
        n_trajectories: int = 100,
        n_steps: int = 30,
        pain_threshold: float = 3.0,
        max_treatments_per_step: int = 3,
        addiction_decay_mode: str = None, doctor_type: str = "SCORE",
) -> dict:
    """Runs single-patient trajectories to clean cross-sections and groupings."""
    metrics = ["Pain", "AddictionScore", "Tolerance", "Craving", "Functionality"]
    extra_metrics = ["PainPreTreatment"]  # tracked for overlay, not given its own subplot row

    all_metrics = metrics + extra_metrics
    profile_trajectories = {prof: {m: [] for m in all_metrics} for prof in PAIN_PROFILES}

    summaries: List[Dict[str, Any]] = []

    for run_idx in range(n_trajectories):

        model = DoctorPatientModel(
            n_patients=1,
            pain_threshold=pain_threshold,
            max_treatments_per_step=max_treatments_per_step,
            addiction_decay_mode=addiction_decay_mode,
            doctor_type=doctor_type,
            seed=run_idx,
        )
        for _ in range(n_steps):
            model.step()

        df = model.datacollector.get_agent_vars_dataframe()
        patient_agent = list(model.agents_by_type[PatientAgent])[0]
        prof = patient_agent.profile_type
        pid = patient_agent.unique_id

        pdata = df.xs(pid, level="AgentID")

        summaries.append({"doctor": doctor_type, "run_idx": run_idx, "profile": prof,
                          **trajectory_summary(pdata)})

        # Optimization Fix: Direct sequence validation out of structured frames
        for m in all_metrics:
            series = pdata[m].to_numpy(dtype=float)
            profile_trajectories[prof][m].append(series)

    grouped_results = {}
    for prof in PAIN_PROFILES:
        grouped_results[prof] = {}
        for m in all_metrics:
            raw_list = profile_trajectories[prof][m]
            if len(raw_list) == 0:
                continue

            arr = np.array(raw_list, dtype=float)
            grouped_results[prof][m] = {
                "mean": np.nanmean(arr, axis=0),
                "p5": np.nanpercentile(arr, 5, axis=0),
                "p25": np.nanpercentile(arr, 25, axis=0),
                "p75": np.nanpercentile(arr, 75, axis=0),
                "p95": np.nanpercentile(arr, 95, axis=0),
                "all": arr,
                "count": len(raw_list)
            }

    return grouped_results, summaries


def plot_batch_by_profile(grouped_results: dict, n_steps: int, base_out_path="dashboard_batch_v8"):
    metrics = ["Pain", "AddictionScore", "Tolerance", "Craving", "Functionality"]
    titles = {
        "Pain": "Pain (0–10)",
        "AddictionScore": "Addiction Score",
        "Tolerance": "Tolerance (potency loss)",
        "Craving": "Craving (psychological)",
        "Functionality": "Functionality (0–10)",
    }
    colors = {
        "Pain": "#E53935",
        "AddictionScore": "#FB8C00",
        "Tolerance": "#5C6BC0",
        "Craving": "#EC407A",
        "Functionality": "#43A047",
    }

    steps = np.arange(n_steps)

    for prof, profile_data in grouped_results.items():
        if not profile_data:
            continue

        sample_size = profile_data[metrics[0]]["count"]
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex=True)
        fig.suptitle(
            f"Monte Carlo Sub-Analysis: {prof} Profile (N={sample_size} Trajectories)\n"
            "Shaded bands: P5–P95 (light) and P25–P75 (dark)",
            fontsize=12, fontweight="bold",
        )

        for ax, m in zip(axes, metrics):
            d = profile_data[m]
            col = colors[m]

            # Background individual trajectories
            for traj in d["all"]:
                ax.plot(steps, traj, color=col, alpha=0.05, lw=0.7)

            ax.fill_between(steps, d["p5"], d["p95"], color=col, alpha=0.10)
            ax.fill_between(steps, d["p25"], d["p75"], color=col, alpha=0.20)
            ax.plot(steps, d["mean"], color=col, lw=2,
                    label="Mean (post-treatment)" if m == "Pain" else "Mean")

            if m == "Pain" and "PainPreTreatment" in profile_data:
                pre = profile_data["PainPreTreatment"]
                ax.plot(steps, pre["mean"], color="#9E9E9E", lw=1.5, ls="--",
                        label="Mean (raw, pre-treatment)")

            if m in ("Functionality", "Pain", "Tolerance", "Craving"):
                ax.set_ylim(0, 10)

            ax.set_ylabel(titles[m], fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.legend(loc="upper right", fontsize=7)

        axes[-1].set_xlabel("Steps")
        plt.tight_layout()

        file_name = f"{base_out_path}_{prof.lower()}.png"
        plt.savefig(file_name, dpi=150)
        print(f"Disaggregated profile plot generated -> {file_name}")
        plt.close()

def report_summaries(rows, csv_path="policy_summary_v9.csv"):
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    for by in (["doctor"], ["profile", "doctor"]):
        g = df.groupby(by)[SUMMARY_METRICS]
        out = g.mean().round(2).astype(str) + " ± " + (1.96 * g.sem()).round(2).astype(str)
        print(out.to_string(), "\n")

    # Paired differences: runs are matched by run_idx (common random numbers)
    print("Paired differences (mean ± 95% CI):")
    pairs = [("SCORE", "RAW_PAIN"), ("FUNCTIONALITY", "RAW_PAIN"), ("FUNCTIONALITY", "SCORE")]
    for m in SUMMARY_METRICS:
        wide = df.pivot(index="run_idx", columns="doctor", values=m)
        for a, b in pairs:
            diff = wide[a] - wide[b]
            print(f"{m:>16}  {a} - {b}: {diff.mean():.2f} ± {1.96 * diff.sem():.2f}")
        print()




if __name__ == "__main__":
    N_STEPS, N_TRAJECTORIES = 30, 100
    all_rows = []
    for dt in ("SCORE", "FUNCTIONALITY", "RAW_PAIN"):
        res, rows = run_batch_by_profile(N_TRAJECTORIES, N_STEPS, 3.0, 3, doctor_type=dt)
        all_rows += rows
        plot_batch_by_profile(res, N_STEPS, base_out_path=f"dashboard_batch_v9_{dt.lower()}")
    report_summaries(all_rows)

