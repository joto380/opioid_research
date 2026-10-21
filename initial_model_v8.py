import numpy as np
import mesa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import random
import math
import json
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────── Constants ────────────────────────

ALPHA: float = 0.2
ADDICTION_PENALTY_WEIGHT: float = 0.3
HIGH_SCORE_THRESHOLD: float = 6.0
MED_SCORE_THRESHOLD:  float = 3.0
ADDICTION_DECAY: float = 0.5

TIER_EFFECTS = {
    "LOW":    {"reduction": 1.5, "addiction_weight": 0.0},
    "MEDIUM": {"reduction": 3.0, "addiction_weight": 1.0},
    "HIGH":   {"reduction": 5.0, "addiction_weight": 3.0},
}

PAIN_PROFILES   = ["SINE", "EXP_DECAY", "LINEAR_INC", "RANDOM_WALK"]
ADDICTION_MODES = ["CONSTANT", "LINEAR", "EXP"]

ADDICTION_AUC_LOW_THRESHOLD:  float = 2.0
ADDICTION_AUC_HIGH_THRESHOLD: float = 5.0

ADDICTION_AUC_EFFECTS = {
    0: {"reduction_penalty": 0.0, "addiction_increase": 0.0},
    1: {"reduction_penalty": 0.5, "addiction_increase": 0.1},
    2: {"reduction_penalty": 1.0, "addiction_increase": 0.2},
}

# ─── Tolerance constants ───────────────────────────────────────────────────────
# Tolerance represents how much the patient has adapted to the drug:
#   - Each MEDIUM/HIGH dose adds tolerance_gain to the score
#   - Tolerance decays slowly between doses (tolerance_decay per step)
#   - High tolerance reduces effective pain reduction (drug loses potency)
#   - Tolerance is bounded [0, 10]
TOLERANCE_GAIN_MEDIUM: float = 0.4   # tolerance added per MEDIUM dose
TOLERANCE_GAIN_HIGH:   float = 0.8   # tolerance added per HIGH dose
TOLERANCE_DECAY_RATE:  float = 0.05  # passive decay per step (slow — body adapts slowly)
TOLERANCE_REDUCTION_FACTOR: float = 0.15  # each point of tolerance reduces drug efficacy by this fraction

# ─── Craving constants ────────────────────────────────────────────────────────
# Craving represents non-pain-driven desire for the drug:
#   - Increases when a dose is received (reward signal)
#   - Increases faster when tolerance is high (chasing the high)
#   - Has a slow natural decay (psychological persistence)
#   - Contributes negatively to functionality independent of pain
CRAVING_GAIN_PER_DOSE:    float = 0.6   # base craving added per addictive dose
CRAVING_TOLERANCE_SCALE:  float = 0.1   # extra craving per unit of tolerance when dosing
CRAVING_DECAY_RATE:       float = 0.08  # natural decay per step
CRAVING_PAIN_SUPPRESSION: float = 0.05  # high pain partially suppresses craving (pain focus)
CRAVING_BOUNDS = (0.0, 10.0)

# ─── Functionality constants ──────────────────────────────────────────────────
# Functionality ∈ [0, 10]: higher = better functioning patient
# It aggregates: pain, addiction_score, tolerance, craving, and treatment tier strength
# Weights control each component's contribution to *impairment*
FUNC_WEIGHT_PAIN:            float = 0.35
FUNC_WEIGHT_ADDICTION:       float = 0.25
FUNC_WEIGHT_TOLERANCE:       float = 0.15
FUNC_WEIGHT_CRAVING:         float = 0.15
FUNC_WEIGHT_TREATMENT_TIER:  float = 0.10   # HIGH=3, MEDIUM=1, LOW=0.3, None=0

TIER_STRENGTH_VALUES = {"HIGH": 3.0, "MEDIUM": 1.0, "LOW": 0.3, None: 0.0}


# ─────────────────────────── Pain Profiles ────────────────────

class PainProfile:
    def __init__(self, profile_type: str, base_pain: float):
        self.profile_type = profile_type
        self.base_pain    = base_pain
        self._rw_current  = base_pain

    def target(self, t: int) -> float:
        if self.profile_type == "SINE":
            return self._sine(t)
        elif self.profile_type == "EXP_DECAY":
            return self._exp_decay(t)
        elif self.profile_type == "LINEAR_INC":
            return self._linear_inc(t)
        elif self.profile_type == "RANDOM_WALK":
            return self._random_walk()
        else:
            raise ValueError(f"Unknown profile type: {self.profile_type}")

    def _sine(self, t):
        return max(0.0, min(10.0, self.base_pain + 2.0 * math.sin(t / 5.0)))

    def _exp_decay(self, t):
        floor = 1.0
        return max(0.0, min(10.0, floor + (self.base_pain - floor) * math.exp(-0.1 * t)))

    def _linear_inc(self, t):
        return max(0.0, min(10.0, self.base_pain + 0.1 * t))

    def _random_walk(self):
        self._rw_current += random.uniform(-0.4, 0.4)
        self._rw_current  = max(0.0, min(10.0, self._rw_current))
        return self._rw_current


# ─────────────────────────── Addiction Dynamics ───────────────

class AddictionDynamic:
    VALID_MODES = {"CONSTANT", "LINEAR", "EXP"}

    def __init__(
        self,
        mode:       str   = "CONSTANT",
        base_rate:  float = ADDICTION_DECAY,
        decay_rate: float = 0.02,
        lambda_:    float = 0.3,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        self.mode       = mode
        self.base_rate  = base_rate
        self.decay_rate = decay_rate
        self.lambda_    = lambda_

    def decay(self, current_score: float, t: int, steps_since_dose: int = 0) -> float:
        if self.mode == "CONSTANT":
            return max(0.0, current_score - self.base_rate)
        elif self.mode == "LINEAR":
            effective_rate = self.base_rate * (1.0 + self.decay_rate * t)
            return max(0.0, current_score - effective_rate)
        elif self.mode == "EXP":
            return current_score * math.exp(-self.lambda_)


# ─────────────────────────── Tolerance Dynamics ───────────────

class ToleranceDynamic:
    """
    Models pharmacological tolerance — the body's adaptation to the drug
    such that the same dose produces a diminished effect over time.

    Per-step passive decay:
        tolerance -= TOLERANCE_DECAY_RATE   (clamped ≥ 0)

    On dose receipt (MEDIUM or HIGH):
        tolerance += gain  (MEDIUM: TOLERANCE_GAIN_MEDIUM, HIGH: TOLERANCE_GAIN_HIGH)

    Effect on treatment:
        effective_reduction = base_reduction * (1 - tolerance * TOLERANCE_REDUCTION_FACTOR)
        i.e. each unit of tolerance reduces drug efficacy by TOLERANCE_REDUCTION_FACTOR (15%).
        At tolerance=6.67 the drug is fully ineffective (reduction→0).
    """

    def decay_step(self, current_tolerance: float) -> float:
        return max(0.0, current_tolerance - TOLERANCE_DECAY_RATE)

    def gain_from_dose(self, tier_name: str) -> float:
        if tier_name == "MEDIUM":
            return TOLERANCE_GAIN_MEDIUM
        elif tier_name == "HIGH":
            return TOLERANCE_GAIN_HIGH
        return 0.0

    def apply_to_reduction(self, base_reduction: float, tolerance: float) -> float:
        """Returns reduced efficacy based on current tolerance level."""
        efficacy_multiplier = max(0.0, 1.0 - tolerance * TOLERANCE_REDUCTION_FACTOR)
        return base_reduction * efficacy_multiplier


# ─────────────────────────── Craving Dynamics ─────────────────

class CravingDynamic:
    """
    Models psychological craving — the non-pain-driven desire for the drug.

    This captures the reward-seeking behaviour that emerges from repeated dosing:
    - Each addictive dose triggers a craving spike proportional to tolerance
      (patients with high tolerance 'chase the high' more intensely)
    - Craving decays naturally each step (but more slowly than addiction score)
    - High pain partially suppresses craving (the patient is focused on pain)

    Per-step update:
        craving = craving * (1 - CRAVING_DECAY_RATE) - pain_suppression * pain
        craving = clamp(craving, 0, 10)

    On dose receipt (MEDIUM or HIGH):
        craving += CRAVING_GAIN_PER_DOSE + tolerance * CRAVING_TOLERANCE_SCALE
    """

    def decay_step(self, current_craving: float, current_pain: float) -> float:
        craving = current_craving * (1.0 - CRAVING_DECAY_RATE)
        craving -= CRAVING_PAIN_SUPPRESSION * current_pain
        return max(CRAVING_BOUNDS[0], min(CRAVING_BOUNDS[1], craving))

    def gain_from_dose(self, tier_name: str, tolerance: float) -> float:
        if tier_name in ("MEDIUM", "HIGH"):
            return CRAVING_GAIN_PER_DOSE + tolerance * CRAVING_TOLERANCE_SCALE
        return 0.0


# ─────────────────────────── Functionality Score ──────────────

def compute_functionality(
    pain:           float,
    addiction:      float,
    tolerance:      float,
    craving:        float,
    current_tier:   Optional[str],
) -> float:
    """
    Functionality ∈ [0, 10] — higher is better.

    Impairment is the weighted sum of all negative factors (each normalised to [0,1]).
    Functionality = 10 * (1 - total_impairment).

    Components of impairment:
      - pain          : normalised over [0, 10]
      - addiction     : normalised over [0, 10]  (can grow beyond 10 in edge cases → clamped)
      - tolerance     : normalised over [0, 10]
      - craving       : normalised over [0, 10]
      - treatment tier: represents iatrogenic burden (HIGH tier = strongest drugs used)
                        normalised over [0, 3]
    """
    tier_strength = TIER_STRENGTH_VALUES.get(current_tier, 0.0)

    impairment = (
        FUNC_WEIGHT_PAIN       * (pain       / 10.0) +
        FUNC_WEIGHT_ADDICTION  * min(1.0, addiction / 10.0) +
        FUNC_WEIGHT_TOLERANCE  * min(1.0, tolerance / 10.0) +
        FUNC_WEIGHT_CRAVING    * (craving    / 10.0) +
        FUNC_WEIGHT_TREATMENT_TIER * (tier_strength / 3.0)
    )
    return max(0.0, min(10.0, 10.0 * (1.0 - impairment)))


# ─────────────────────────── Treatment Tier Logic ─────────────

def compute_treatment_tier(pain: float, addiction_score: float, pain_threshold: float) -> str | None:
    if pain < pain_threshold:
        return None
    treatment_score = pain - (addiction_score * ADDICTION_PENALTY_WEIGHT)
    if treatment_score >= HIGH_SCORE_THRESHOLD:
        return "HIGH"
    elif treatment_score >= MED_SCORE_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


def addiction_auc_zone(addiction_auc: float, current_step: int) -> int:
    if current_step == 0:
        return 0
    avg = addiction_auc / current_step
    if avg < ADDICTION_AUC_LOW_THRESHOLD:
        return 0
    elif avg < ADDICTION_AUC_HIGH_THRESHOLD:
        return 1
    else:
        return 2


# ─────────────────────────── Agents ───────────────────────────

class PatientAgent(mesa.Agent):
    def __init__(
        self,
        model,
        base_pain:            float = 5.0,
        addiction_decay_mode: str   = "CONSTANT",
    ):
        super().__init__(model)
        self.base_pain   = base_pain
        self.pain: float = base_pain
        self.pain_before_treatment: float = self.base_pain

        profile_type      = random.choice(PAIN_PROFILES)
        self.profile      = PainProfile(profile_type, base_pain)
        self.profile_type = profile_type

        self.addiction_dynamic  = AddictionDynamic(mode=addiction_decay_mode)
        self.tolerance_dynamic  = ToleranceDynamic()
        self.craving_dynamic    = CravingDynamic()

        self.addiction_score_before_treatment: float = 0.0

        self.treated_this_step: bool  = False
        self.current_tier: str | None = None

        self.medium_treatment_count: int = 0
        self.high_treatment_count:   int = 0
        self.addiction_score:      float = 0.0
        self.downgrades_received:    int = 0

        # ── NEW: Tolerance & Craving ───────────────────────────────────────
        self.tolerance: float = 0.0
        self.craving:   float = 0.0
        # ─────────────────────────────────────────────────────────────────

        # ── Functionality ─────────────────────────────────────────────────
        self.functionality: float = 10.0
        # ─────────────────────────────────────────────────────────────────

        self.steps_since_last_dose: int = 0

        self._prev_pain: float = base_pain
        self.pain_auc:   float = 0.0

        self._prev_addiction_score: float = 0.0
        self.addiction_auc:         float = 0.0

        self.profile_targets: list[float] = []

    def _evolve_pain(self, target: float) -> float:
        noise = random.uniform(-0.3, 0.3)
        return max(0.0, min(10.0, ALPHA * self.pain + (1.0 - ALPHA) * target + noise))

    def step(self):
        t      = self.model.current_step
        target = self.profile.target(t)
        self.profile_targets.append(target)
        self.pain = self._evolve_pain(target)

        self.pain_auc  += 0.5 * (self._prev_pain + self.pain)
        self._prev_pain = self.pain

        self.pain_before_treatment = self.pain
        self.addiction_score_before_treatment = self.addiction_score

        self.treated_this_step = False
        self.current_tier      = None

        self.steps_since_last_dose += 1

        # Addiction decay
        self.addiction_score = self.addiction_dynamic.decay(
            self.addiction_score,
            self.model.current_step,
            self.steps_since_last_dose,
        )

        # ── NEW: Tolerance passive decay ──────────────────────────────────
        self.tolerance = self.tolerance_dynamic.decay_step(self.tolerance)
        # ─────────────────────────────────────────────────────────────────

        # ── NEW: Craving passive decay (pain-suppressed) ──────────────────
        self.craving = self.craving_dynamic.decay_step(self.craving, self.pain)
        # ─────────────────────────────────────────────────────────────────

        self.addiction_auc         += 0.5 * (self._prev_addiction_score + self.addiction_score)
        self._prev_addiction_score  = self.addiction_score

        # ── Update functionality (pre-treatment snapshot for this step) ───
        self.functionality = compute_functionality(
            self.pain, self.addiction_score, self.tolerance, self.craving, None
        )
        # ─────────────────────────────────────────────────────────────────

    def receive_treatment(self, tier_name: str):
        effects          = TIER_EFFECTS[tier_name]
        reduction        = effects["reduction"]
        addiction_weight = effects["addiction_weight"]

        # Addiction-AUC zone modifiers
        zone    = addiction_auc_zone(self.addiction_auc, self.model.current_step)
        zone_fx = ADDICTION_AUC_EFFECTS[zone]
        if tier_name in ("MEDIUM", "HIGH"):
            reduction        = max(0.0, reduction - zone_fx["reduction_penalty"])
            addiction_weight = addiction_weight + zone_fx["addiction_increase"]

        # ── NEW: Apply tolerance to reduction ─────────────────────────────
        if tier_name in ("MEDIUM", "HIGH"):
            effective_reduction = self.tolerance_dynamic.apply_to_reduction(reduction, self.tolerance)
        else:
            effective_reduction = reduction
        # ─────────────────────────────────────────────────────────────────

        self.pain = max(0.0, self.pain - effective_reduction)
        self.treated_this_step = True
        self.current_tier      = tier_name

        if tier_name == "MEDIUM":
            self.medium_treatment_count += 1
        elif tier_name == "HIGH":
            self.high_treatment_count += 1

        self.addiction_score += addiction_weight

        # ── NEW: Tolerance gain on dose ───────────────────────────────────
        self.tolerance = min(10.0, self.tolerance + self.tolerance_dynamic.gain_from_dose(tier_name))
        # ─────────────────────────────────────────────────────────────────

        # ── NEW: Craving gain on dose ─────────────────────────────────────
        self.craving = min(
            CRAVING_BOUNDS[1],
            self.craving + self.craving_dynamic.gain_from_dose(tier_name, self.tolerance)
        )
        # ─────────────────────────────────────────────────────────────────

        if tier_name in ("MEDIUM", "HIGH"):
            self.steps_since_last_dose = 0

        # Update functionality post-treatment
        self.functionality = compute_functionality(
            self.pain, self.addiction_score, self.tolerance, self.craving, tier_name
        )


class DoctorAgent(mesa.Agent):
    def __init__(self, model, pain_threshold: float = 3.0, max_treatments_per_step: int = 2):
        super().__init__(model)
        self.pain_threshold          = pain_threshold
        self.max_treatments_per_step = max_treatments_per_step
        self.treatments_given:  int  = 0
        self.downgrades_issued: int  = 0

    def step(self):
        treatments_this_step = 0
        patients = list(self.model.agents_by_type[PatientAgent])
        patients.sort(key=lambda p: p.pain, reverse=True)

        for patient in patients:
            if treatments_this_step >= self.max_treatments_per_step:
                break

            raw_tier    = compute_treatment_tier(patient.pain, 0.0, self.pain_threshold)
            chosen_tier = compute_treatment_tier(patient.pain, patient.addiction_score, self.pain_threshold)

            if chosen_tier is None:
                continue

            tier_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            if raw_tier is not None and tier_order[chosen_tier] < tier_order[raw_tier]:
                self.downgrades_issued      += 1
                patient.downgrades_received += 1

            patient.receive_treatment(chosen_tier)
            treatments_this_step  += 1
            self.treatments_given += 1


# ─────────────────────────── Model ────────────────────────────

class DoctorPatientModel(mesa.Model):
    def __init__(
        self,
        n_patients:              int   = 3,
        pain_threshold:          float = 3.0,
        max_treatments_per_step: int   = 2,
        addiction_decay_mode:    str   = "CONSTANT",
    ):
        super().__init__()
        self.current_step = 0

        for _ in range(n_patients):
            base_pain = random.uniform(3.0, 8.0)
            mode = (
                random.choice(ADDICTION_MODES)
                if addiction_decay_mode is None
                else addiction_decay_mode
            )
            PatientAgent(model=self, base_pain=base_pain, addiction_decay_mode=mode)

        DoctorAgent(
            model=self,
            pain_threshold=pain_threshold,
            max_treatments_per_step=max_treatments_per_step,
        )

        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Pain":             lambda a: a.pain                   if isinstance(a, PatientAgent) else None,
                "Tier":             lambda a: a.current_tier           if isinstance(a, PatientAgent) else None,
                "AddictionScore":   lambda a: a.addiction_score        if isinstance(a, PatientAgent) else None,
                "TreatmentScore": lambda a: (
                    a.pain_before_treatment - a.addiction_score_before_treatment * ADDICTION_PENALTY_WEIGHT
                )                                                      if isinstance(a, PatientAgent) else None,
                "ProfileType":      lambda a: a.profile_type           if isinstance(a, PatientAgent) else None,
                "AddictionMode":    lambda a: a.addiction_dynamic.mode if isinstance(a, PatientAgent) else None,
                "PainAUC":          lambda a: a.pain_auc               if isinstance(a, PatientAgent) else None,
                "AddictionAUC":     lambda a: a.addiction_auc          if isinstance(a, PatientAgent) else None,
                "AddictionAUCZone": lambda a: (
                    addiction_auc_zone(a.addiction_auc, a.model.current_step)
                )                                                      if isinstance(a, PatientAgent) else None,
                # ── NEW reporters ─────────────────────────────────────────
                "Tolerance":        lambda a: a.tolerance              if isinstance(a, PatientAgent) else None,
                "Craving":          lambda a: a.craving                if isinstance(a, PatientAgent) else None,
                "Functionality":    lambda a: a.functionality          if isinstance(a, PatientAgent) else None,
                # ─────────────────────────────────────────────────────────
                "MediumTreatments": lambda a: a.medium_treatment_count if isinstance(a, PatientAgent) else None,
                "HighTreatments":   lambda a: a.high_treatment_count   if isinstance(a, PatientAgent) else None,
                "Downgrades":       lambda a: a.downgrades_received    if isinstance(a, PatientAgent) else None,
                "Treated":          lambda a: a.treated_this_step      if isinstance(a, PatientAgent) else None,
                "TotalTreatments":  lambda a: a.treatments_given       if isinstance(a, DoctorAgent)  else None,
                "TotalDowngrades":  lambda a: a.downgrades_issued      if isinstance(a, DoctorAgent)  else None,
            }
        )

    def step(self):
        self.agents_by_type[PatientAgent].do("step")
        self.agents_by_type[DoctorAgent].do("step")
        self.current_step += 1
        self.datacollector.collect(self)


# ─────────────────────────── Colours ──────────────────────────

PROFILE_COLORS = {
    "SINE":        "#1565C0",
    "EXP_DECAY":   "#2E7D32",
    "LINEAR_INC":  "#B71C1C",
    "RANDOM_WALK": "#6A1B9A",
}
TIER_COLORS = {
    "LOW":    "#4CAF50",
    "MEDIUM": "#FF9800",
    "HIGH":   "#F44336",
    None:     "#AAAAAA",
}
ADDICTION_MODE_COLORS = {
    "CONSTANT": "#1565C0",
    "LINEAR":   "#E65100",
    "EXP":      "#1B5E20",
}
ADDICTION_AUC_ZONE_COLORS = {
    0: "#4CAF50",
    1: "#FF9800",
    2: "#F44336",
}


# ─────────────────────────── Single-run Plot ──────────────────

def plot_single_run(model, df, n_steps, out_path="simulation_plots_v8.png"):
    patient_ids  = [a.unique_id for a in model.agents_by_type[PatientAgent]]
    agent_lookup = {a.unique_id: a for a in model.agents_by_type[PatientAgent]}
    n_patients   = len(patient_ids)
    N_ROWS       = 6   # pain, addiction, treatment-score, addiction-AUC, tolerance+craving, functionality

    fig, axes = plt.subplots(
        nrows=N_ROWS, ncols=n_patients,
        figsize=(6 * n_patients, 20),
        sharex=True,
    )
    if n_patients == 1:
        axes = np.array(axes).reshape(N_ROWS, 1)

    fig.suptitle(
        "Doctor-Patient Simulation v8 — Tolerance · Craving · Functionality\n"
        f"α={ALPHA}  |  tol_decay={TOLERANCE_DECAY_RATE}/step  |  "
        f"craving_decay={CRAVING_DECAY_RATE}/step  |  steps={n_steps}",
        fontsize=12, fontweight="bold",
    )

    for col, pid in enumerate(patient_ids):
        ax_pain       = axes[0, col]
        ax_addict     = axes[1, col]
        ax_tscore     = axes[2, col]
        ax_addict_auc = axes[3, col]
        ax_tol_crav   = axes[4, col]   # NEW: tolerance + craving overlay
        ax_func       = axes[5, col]   # NEW: functionality

        pdata = df.xs(pid, level="AgentID") if pid in df.index.get_level_values("AgentID") else None
        if pdata is None or pdata.empty:
            continue

        steps          = pdata.index.get_level_values("Step").tolist()
        pain_vals      = pdata["Pain"].tolist()
        tier_vals      = pdata["Tier"].tolist()
        addict_vals    = pdata["AddictionScore"].tolist()
        tscore_vals    = pdata["TreatmentScore"].tolist()
        addict_auc_vals  = pdata["AddictionAUC"].tolist()
        addict_zone_vals = pdata["AddictionAUCZone"].tolist()
        tol_vals       = pdata["Tolerance"].tolist()
        crav_vals      = pdata["Craving"].tolist()
        func_vals      = pdata["Functionality"].tolist()

        auc_final   = pdata["PainAUC"].dropna().iloc[-1]   if not pdata["PainAUC"].dropna().empty   else 0.0
        profile     = pdata["ProfileType"].dropna().iloc[0] if not pdata["ProfileType"].dropna().empty else "?"
        addict_mode = pdata["AddictionMode"].dropna().iloc[0] if not pdata["AddictionMode"].dropna().empty else "?"

        profile_color = PROFILE_COLORS.get(profile, "#333333")
        addict_color  = ADDICTION_MODE_COLORS.get(addict_mode, "#555555")

        agent           = agent_lookup[pid]
        raw_targets     = agent.profile_targets
        aligned_targets = [agent.base_pain] + raw_targets[: len(steps) - 1]

        # ── 1. Pain ─────────────────────────────────────────────────────
        ax_pain.plot(steps, aligned_targets, color=profile_color, lw=1.6,
                     ls="--", alpha=0.55, zorder=1, label="Profile target")
        ax_pain.plot(steps, pain_vals, color=profile_color, lw=1.2, alpha=0.4, zorder=2)
        for i in range(len(steps) - 1):
            ax_pain.plot(steps[i:i+2], pain_vals[i:i+2],
                         color=TIER_COLORS.get(tier_vals[i], "#888"), lw=2.2, zorder=3)
        ax_pain.scatter(steps, pain_vals,
                        c=[TIER_COLORS.get(t, "#888") for t in tier_vals], s=40, zorder=5)
        ax_pain.fill_between(steps, pain_vals, alpha=0.07, color=profile_color)
        ax_pain.axhspan(0, 3, color="#4CAF50", alpha=0.06)
        ax_pain.axhspan(3, 6, color="#FF9800", alpha=0.06)
        ax_pain.axhspan(6, 10, color="#F44336", alpha=0.06)
        ax_pain.set_ylim(0, 10)
        ax_pain.set_ylabel("Pain (0–10)")
        ax_pain.set_title(f"Patient {pid} — {profile}\nPain & Tier  |  AUC = {auc_final:.1f}")
        tier_patches = [mpatches.Patch(color=c, label=t) for t, c in TIER_COLORS.items() if t]
        ax_pain.legend(handles=tier_patches, fontsize=7, loc="upper right")

        # ── 2. Addiction Score ──────────────────────────────────────────
        ax_addict.plot(steps, addict_vals, color=addict_color, lw=2.0, label=f"mode: {addict_mode}")
        ax_addict.fill_between(steps, addict_vals, alpha=0.15, color=addict_color)
        ax_addict.set_ylabel("Addiction score")
        ax_addict.set_title(f"Addiction Score ({addict_mode})")
        ax_addict.legend(fontsize=7, loc="upper right")

        # ── 3. Treatment Score ──────────────────────────────────────────
        ax_tscore.plot(steps, tscore_vals, color=profile_color, lw=2.0)
        ax_tscore.fill_between(steps, tscore_vals, alpha=0.12, color=profile_color)
        ax_tscore.axhline(HIGH_SCORE_THRESHOLD, color="#F44336", lw=1.2, ls="--",
                          label=f"HIGH ({HIGH_SCORE_THRESHOLD})")
        ax_tscore.axhline(MED_SCORE_THRESHOLD,  color="#FF9800", lw=1.2, ls="--",
                          label=f"MED ({MED_SCORE_THRESHOLD})")
        ax_tscore.set_ylabel("Treatment score")
        ax_tscore.set_title("Treatment Score")
        ax_tscore.legend(fontsize=7, loc="upper right")

        # ── 4. Addiction AUC & Zone ─────────────────────────────────────
        low_line  = [ADDICTION_AUC_LOW_THRESHOLD  * s for s in steps]
        high_line = [ADDICTION_AUC_HIGH_THRESHOLD * s for s in steps]
        for i in range(len(steps) - 1):
            ax_addict_auc.plot(steps[i:i+2], addict_auc_vals[i:i+2],
                               color=ADDICTION_AUC_ZONE_COLORS.get(addict_zone_vals[i], "#888"),
                               lw=2.4, zorder=3)
        ax_addict_auc.scatter(steps, addict_auc_vals,
                              c=[ADDICTION_AUC_ZONE_COLORS.get(z, "#888") for z in addict_zone_vals],
                              s=30, zorder=5)
        ax_addict_auc.plot(steps, low_line,  color="#FF9800", lw=1.2, ls="--", alpha=0.7)
        ax_addict_auc.plot(steps, high_line, color="#F44336", lw=1.2, ls="--", alpha=0.7)
        ax_addict_auc.set_ylabel("Addiction AUC")
        ax_addict_auc.set_title("Addiction AUC & Zone")

        # ── 5. NEW: Tolerance & Craving ─────────────────────────────────
        ax_tol_crav.plot(steps, tol_vals,  color="#5C6BC0", lw=2.2, label="Tolerance")
        ax_tol_crav.fill_between(steps, tol_vals,  alpha=0.12, color="#5C6BC0")
        ax_tol_crav.plot(steps, crav_vals, color="#EC407A", lw=2.2, label="Craving",  ls="--")
        ax_tol_crav.fill_between(steps, crav_vals, alpha=0.10, color="#EC407A")
        ax_tol_crav.set_ylim(0, 10)
        ax_tol_crav.set_ylabel("Score (0–10)")
        ax_tol_crav.set_title("Tolerance & Craving")
        ax_tol_crav.legend(fontsize=8, loc="upper right")

        # ── 6. NEW: Functionality ────────────────────────────────────────
        # Colour segments by functionality zone (high=green, low=red)
        func_cmap = plt.cm.RdYlGn
        for i in range(len(steps) - 1):
            fval = (func_vals[i] / 10.0)
            ax_func.plot(steps[i:i+2], func_vals[i:i+2],
                         color=func_cmap(fval), lw=2.5, zorder=3)
        ax_func.scatter(steps, func_vals,
                        c=[func_cmap(v / 10.0) for v in func_vals], s=40, zorder=5)
        ax_func.fill_between(steps, func_vals, alpha=0.08, color="#66BB6A")
        ax_func.axhspan(0,  4, color="#F44336", alpha=0.05)
        ax_func.axhspan(4,  7, color="#FF9800", alpha=0.05)
        ax_func.axhspan(7, 10, color="#4CAF50", alpha=0.05)
        ax_func.set_ylim(0, 10)
        ax_func.set_ylabel("Functionality (0–10)")
        ax_func.set_xlabel("Step")
        ax_func.set_title("Functionality Score\n(higher = better)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSingle-run plot saved → {out_path}")
    plt.close()


# ─────────────────────────── Batch / Monte-Carlo Plot ─────────

def run_batch(
    n_trajectories:          int   = 50,
    n_steps:                 int   = 30,
    n_patients:              int   = 1,
    pain_threshold:          float = 3.0,
    max_treatments_per_step: int   = 3,
    addiction_decay_mode:    str   = None,
) -> dict:
    """
    Runs `n_trajectories` independent simulations of `n_steps` each and
    collects per-step statistics across all patients across all runs.

    Returns a dict of arrays shaped (n_steps,) with mean, p5, p95 for each metric.
    """
    metrics = ["Pain", "AddictionScore", "Tolerance", "Craving", "Functionality"]
    # shape: (n_trajectories * n_patients, n_steps)
    trajectories = {m: [] for m in metrics}

    for run_idx in range(n_trajectories):
        model = DoctorPatientModel(
            n_patients=n_patients,
            pain_threshold=pain_threshold,
            max_treatments_per_step=max_treatments_per_step,
            addiction_decay_mode=addiction_decay_mode,
        )
        for _ in range(n_steps):
            model.step()

        df = model.datacollector.get_agent_vars_dataframe()
        patient_df = df[df["Pain"].notna()]

        for pid in [a.unique_id for a in model.agents_by_type[PatientAgent]]:
            try:
                pdata = patient_df.xs(pid, level="AgentID")
            except KeyError:
                continue
            steps_avail = sorted(pdata.index.get_level_values("Step").tolist())
            for m in metrics:
                vals = []
                for s in range(n_steps):
                    if s in steps_avail:
                        val = pdata.loc[s, m] if s in pdata.index.get_level_values("Step") else np.nan
                    else:
                        val = np.nan
                    if isinstance(val, (list, np.ndarray)):
                        val = float(val[0]) if len(val) > 0 else np.nan
                    vals.append(float(val) if val is not None else np.nan)
                trajectories[m].append(vals)

    results = {}
    for m in metrics:
        arr = np.array(trajectories[m], dtype=float)
        results[m] = {
            "mean": np.nanmean(arr, axis=0),
            "p5":   np.nanpercentile(arr, 5,  axis=0),
            "p25":  np.nanpercentile(arr, 25, axis=0),
            "p75":  np.nanpercentile(arr, 75, axis=0),
            "p95":  np.nanpercentile(arr, 95, axis=0),
            "all":  arr,
        }
    return results


def plot_batch(results: dict, n_steps: int, n_trajectories: int,
               out_path="simulation_batch_v8.png"):
    metrics = ["Pain", "AddictionScore", "Tolerance", "Craving", "Functionality"]
    titles  = {
        "Pain":           "Pain (0–10)",
        "AddictionScore": "Addiction Score",
        "Tolerance":      "Tolerance (drug potency loss)",
        "Craving":        "Craving (non-pain desire)",
        "Functionality":  "Functionality (0–10, higher=better)",
    }
    colors  = {
        "Pain":           "#E53935",
        "AddictionScore": "#FB8C00",
        "Tolerance":      "#5C6BC0",
        "Craving":        "#EC407A",
        "Functionality":  "#43A047",
    }

    steps = np.arange(n_steps)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)), sharex=True)

    fig.suptitle(
        f"Monte Carlo Dashboard — {n_trajectories} Trajectories × {n_steps} Steps\n"
        "Shaded bands: P5–P25 (light) and P25–P75 (dark)",
        fontsize=13, fontweight="bold",
    )

    for ax, m in zip(axes, metrics):
        d   = results[m]
        col = colors[m]

        # All individual trajectories (very faint)
        for traj in d["all"]:
            ax.plot(steps, traj, color=col, alpha=0.04, lw=0.7)

        # Percentile bands
        ax.fill_between(steps, d["p5"],  d["p95"], color=col, alpha=0.10, label="P5–P95")
        ax.fill_between(steps, d["p25"], d["p75"], color=col, alpha=0.22, label="P25–P75")

        # Mean
        ax.plot(steps, d["mean"], color=col, lw=2.5, label="Mean", zorder=5)

        if m == "Functionality":
            ax.set_ylim(0, 10)
            ax.axhspan(0, 4,  color="#F44336", alpha=0.05)
            ax.axhspan(4, 7,  color="#FF9800", alpha=0.05)
            ax.axhspan(7, 10, color="#4CAF50", alpha=0.05)
        elif m in ("Pain", "Tolerance", "Craving"):
            ax.set_ylim(0, 10)

        ax.set_ylabel(titles[m], fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Batch plot saved → {out_path}")
    plt.close()


# ─────────────────────────── Dashboard ───────────────────────
# Combines both modes into a side-by-side summary dashboard figure.

def run_dashboard(
    mode:                    str   = "single",   # "single" | "batch"
    n_steps:                 int   = 30,
    n_patients_single:       int   = 3,
    n_trajectories:          int   = 50,
    pain_threshold:          float = 3.0,
    max_treatments_per_step: int   = 3,
    addiction_decay_mode:    str   = None,
):
    print(f"\n=== Simulation v8 Dashboard — mode={mode} ===\n")

    if mode == "single":
        model = DoctorPatientModel(
            n_patients=n_patients_single,
            pain_threshold=pain_threshold,
            max_treatments_per_step=max_treatments_per_step,
            addiction_decay_mode=addiction_decay_mode,
        )
        for _ in range(n_steps):
            model.step()
        df = model.datacollector.get_agent_vars_dataframe()
        plot_single_run(model, df, n_steps, out_path="dashboard_single_v8.png")

        # Summary
        print("\n=== Per-Patient Summary ===")
        for a in model.agents_by_type[PatientAgent]:
            final_zone = addiction_auc_zone(a.addiction_auc, model.current_step)
            print(
                f"Patient {a.unique_id:>2} | profile={a.profile_type:<12} "
                f"decay={a.addiction_dynamic.mode:<8} | "
                f"medium={a.medium_treatment_count:>2}  high={a.high_treatment_count:>2}  "
                f"final_pain={a.pain:>5.2f}  addiction={a.addiction_score:>5.2f}  "
                f"tolerance={a.tolerance:>5.2f}  craving={a.craving:>5.2f}  "
                f"functionality={a.functionality:>5.2f}  zone={final_zone}"
            )

    elif mode == "batch":
        print(f"Running {n_trajectories} trajectories × {n_steps} steps …")
        results = run_batch(
            n_trajectories=n_trajectories,
            n_steps=n_steps,
            n_patients=1,
            pain_threshold=pain_threshold,
            max_treatments_per_step=max_treatments_per_step,
            addiction_decay_mode=addiction_decay_mode,
        )
        plot_batch(results, n_steps, n_trajectories, out_path="dashboard_batch_v8.png")

        # Print final-step statistics
        print("\n=== Final-Step Statistics (step %d) ===" % (n_steps - 1))
        for m in ["Pain", "AddictionScore", "Tolerance", "Craving", "Functionality"]:
            d = results[m]
            print(
                f"  {m:<18}: mean={d['mean'][-1]:5.2f}  "
                f"P5={d['p5'][-1]:5.2f}  P25={d['p25'][-1]:5.2f}  "
                f"P75={d['p75'][-1]:5.2f}  P95={d['p95'][-1]:5.2f}"
            )

    else:
        raise ValueError(f"mode must be 'single' or 'batch', got {mode!r}")


# ─────────────────────────── Run ──────────────────────────────

if __name__ == "__main__":
    import sys

    # Default: run single mode then batch mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("single", "both"):
        run_dashboard(
            mode="single",
            n_steps=30,
            n_patients_single=3,
            max_treatments_per_step=3,
            addiction_decay_mode=None,
        )

    if mode in ("batch", "both"):
        run_dashboard(
            mode="batch",
            n_steps=30,
            n_trajectories=75,
            max_treatments_per_step=3,
            addiction_decay_mode=None,
        )