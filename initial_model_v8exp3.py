import numpy as np
import mesa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import math
import sys
from typing import Optional, List, Dict, Any

# ─────────────────────────── CONSTANTS & CONFIG ────────────────────────

ALPHA: float = 0.2
ADDICTION_PENALTY_WEIGHT: float = 0.3
HIGH_SCORE_THRESHOLD: float = 6.0
MED_SCORE_THRESHOLD: float = 3.0

# AUC Dynamic Memory Window (Fixes lifetime time-step dilution)
AUC_MEM_WINDOW: int = 10

TIER_EFFECTS = {
    "LOW": {"reduction": 1.5, "addiction_weight": 0.0},
    "MEDIUM": {"reduction": 3.0, "addiction_weight": 1.0},
    "HIGH": {"reduction": 5.0, "addiction_weight": 3.0},
}

PAIN_PROFILES = ["SINE", "EXP_DECAY", "LINEAR_INC", "RANDOM_WALK"]
ADDICTION_MODES = ["CONSTANT", "LINEAR", "EXP"]

ADDICTION_AUC_LOW_THRESHOLD: float = 2.0
ADDICTION_AUC_HIGH_THRESHOLD: float = 5.0

ADDICTION_AUC_EFFECTS = {
    0: {"reduction_penalty": 0.0, "addiction_increase": 0.0},
    1: {"reduction_penalty": 0.5, "addiction_increase": 0.1},
    2: {"reduction_penalty": 1.0, "addiction_increase": 0.2},
}

# Pain "override" dynamics: above PAIN_CRITICAL_THRESHOLD, the pain term in the
# treatment score explodes (power-law) so that no realistic addiction-risk or
# tolerance penalty can pull effective_pain_score back below HIGH_SCORE_THRESHOLD.
# Models acute/breakthrough pain clinically overriding addiction-risk caution.
PAIN_CRITICAL_THRESHOLD: float = 8.0
PAIN_EXPLOSION_COEF: float = 5.0
PAIN_EXPLOSION_POWER: float = 3.0

# Stochastic volatility injected into addiction risk each step (mean 0).
ADDICTION_RISK_NOISE_STD: float = 0.15

# Tolerance constants (Upgraded to handle Sigmoidal/Emax logic)
TOLERANCE_GAIN_MEDIUM: float = 0.4
TOLERANCE_GAIN_HIGH: float = 0.8
TOLERANCE_DECAY_RATE: float = 0.05
TOLERANCE_50: float = 5.0  # Inflection point (Km) for sigmoidal efficacy loss

# Craving constants
CRAVING_GAIN_PER_DOSE: float = 0.6
CRAVING_TOLERANCE_SCALE: float = 0.1
CRAVING_DECAY_RATE: float = 0.08
CRAVING_PAIN_SUPPRESSION: float = 0.05
CRAVING_BOUNDS = (0.0, 10.0)

# Functionality weights
FUNC_WEIGHT_PAIN: float = 0.35
FUNC_WEIGHT_ADDICTION: float = 0.25
FUNC_WEIGHT_TOLERANCE: float = 0.15
FUNC_WEIGHT_CRAVING: float = 0.15
FUNC_WEIGHT_TREATMENT_TIER: float = 0.10

TIER_STRENGTH_VALUES = {"HIGH": 3.0, "MEDIUM": 1.0, "LOW": 0.3, None: 0.0}

#Pain convexity
PAIN_CONVEX_COEF: float = 12.0   # strength of the convex term
PAIN_CONVEX_POWER: float = 4.0   # curvature; higher = more weight concentrated at the top


# ─────────────────────────── COMPONENT DYNAMICS ────────────────────────

class PainProfile:
    def __init__(self, profile_type: str, base_pain: float):
        self.profile_type = profile_type.upper()
        self.base_pain = base_pain
        self._rw_current = base_pain

    def target(self, t: int) -> float:
        if self.profile_type == "SINE":
            return max(0.0, min(10.0, self.base_pain + 2.0 * math.sin(t / 5.0)))
        elif self.profile_type == "EXP_DECAY":
            return max(0.0, min(10.0, 1.0 + (self.base_pain - 1.0) * math.exp(-0.1 * t)))
        elif self.profile_type == "LINEAR_INC":
            return max(0.0, min(10.0, self.base_pain + 0.1 * t))
        elif self.profile_type == "RANDOM_WALK":
            self._rw_current += random.uniform(-0.4, 0.4)
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

""" def compute_pain_component(pain: float) -> float:
    
    #Smooth convex transform of raw pain for the doctor's treatment score.
    #f(p) = p + A * (p/10)^k   -> monotone, convex, no kink.
    #Low pain stays ~linear; high pain gets a disproportionately large score,
    #so the addiction-risk and tolerance penalties are increasingly outweighed.
    
    return pain + PAIN_CONVEX_COEF * (pain / 10.0) ** PAIN_CONVEX_POWER """



def compute_functionality(pain: float, addiction: float, tolerance: float, craving: float,
                          current_tier: Optional[str]) -> float:
    tier_strength = TIER_STRENGTH_VALUES.get(current_tier, 0.0)
    impairment = (
            FUNC_WEIGHT_PAIN * (pain / 10.0) +
            FUNC_WEIGHT_ADDICTION * min(1.0, addiction / 10.0) +
            FUNC_WEIGHT_TOLERANCE * min(1.0, tolerance / 10.0) +
            FUNC_WEIGHT_CRAVING * (craving / 10.0) +
            FUNC_WEIGHT_TREATMENT_TIER * (tier_strength / 3.0)
    )
    return max(0.0, min(10.0, 10.0 * (1.0 - impairment)))


# ─────────────────────────── AGENTS ────────────────────────

class PatientAgent(mesa.Agent):
    def __init__(self, model, base_pain: float = 5.0, addiction_decay_mode: str = "CONSTANT"):
        super().__init__(model)
        self.base_pain = base_pain
        self.pain = base_pain

        self.profile_type = random.choice(PAIN_PROFILES)
        self.profile = PainProfile(self.profile_type, base_pain)

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

        noise = random.uniform(-0.3, 0.3)
        self.pain = max(0.0, min(10.0, ALPHA * self.pain + (1.0 - ALPHA) * target + noise))

        self.steps_since_last_dose += 1
        self.addiction_score = self.addiction_dynamic.decay(self.addiction_score, t)
        # Volatility: addiction risk doesn't move purely on rails - add mean-0
        # Gaussian noise each step, floored at 0 since risk can't go negative.
        self.addiction_score = max(0.0, self.addiction_score + random.gauss(0.0, ADDICTION_RISK_NOISE_STD))
        self.tolerance = self.tolerance_dynamic.decay_step(self.tolerance)
        self.craving = self.craving_dynamic.decay_step(self.craving, self.pain)

        self.treated_this_step = False
        self.current_tier = None

    def receive_treatment(self, tier_name: str):
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

        self.pain = max(0.0, self.pain - effective_reduction)
        self.treated_this_step = True
        self.current_tier = tier_name

        if tier_name == "MEDIUM":
            self.medium_treatment_count += 1
        elif tier_name == "HIGH":
            self.high_treatment_count += 1

        self.addiction_score += addiction_weight
        self.tolerance = min(10.0, self.tolerance + self.tolerance_dynamic.gain_from_dose(tier_name))
        self.craving = min(CRAVING_BOUNDS[1],
                           self.craving + self.craving_dynamic.gain_from_dose(tier_name, self.tolerance))

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


# ─────────────────────────── MODEL ENVIRONMENT ────────────────────────

class DoctorPatientModel(mesa.Model):
    def __init__(self, n_patients: int = 1, pain_threshold: float = 3.0, max_treatments_per_step: int = 2,
                 addiction_decay_mode: str = None):
        super().__init__()
        self.current_step = 0

        for _ in range(n_patients):
            base_pain = random.uniform(3.0, 8.0)
            mode = random.choice(ADDICTION_MODES) if addiction_decay_mode is None else addiction_decay_mode
            PatientAgent(model=self, base_pain=base_pain, addiction_decay_mode=mode)

        DoctorAgent(model=self, pain_threshold=pain_threshold, max_treatments_per_step=max_treatments_per_step)

        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Pain": lambda a: getattr(a, 'pain', None) if isinstance(a, PatientAgent) else None,
                "Tier": lambda a: getattr(a, 'current_tier', None) if isinstance(a, PatientAgent) else None,
                "AddictionScore": lambda a: getattr(a, 'addiction_score', None) if isinstance(a,
                                                                                              PatientAgent) else None,
                "Tolerance": lambda a: getattr(a, 'tolerance', None) if isinstance(a, PatientAgent) else None,
                "Craving": lambda a: getattr(a, 'craving', None) if isinstance(a, PatientAgent) else None,
                "Functionality": lambda a: getattr(a, 'functionality', None) if isinstance(a, PatientAgent) else None,
                "ProfileType": lambda a: getattr(a, 'profile_type', None) if isinstance(a, PatientAgent) else None,
            }
        )

    def step(self):
        patients = self.agents_by_type[PatientAgent]
        doctors = self.agents_by_type[DoctorAgent]

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
        addiction_decay_mode: str = None,
) -> dict:
    """Runs single-patient trajectories to clean cross-sections and groupings."""
    metrics = ["Pain", "AddictionScore", "Tolerance", "Craving", "Functionality"]
    profile_trajectories = {prof: {m: [] for m in metrics} for prof in PAIN_PROFILES}

    for run_idx in range(n_trajectories):
        model = DoctorPatientModel(
            n_patients=1,
            pain_threshold=pain_threshold,
            max_treatments_per_step=max_treatments_per_step,
            addiction_decay_mode=addiction_decay_mode,
        )
        for _ in range(n_steps):
            model.step()

        df = model.datacollector.get_agent_vars_dataframe()
        patient_agent = list(model.agents_by_type[PatientAgent])[0]
        prof = patient_agent.profile_type
        pid = patient_agent.unique_id

        pdata = df.xs(pid, level="AgentID")

        # Optimization Fix: Direct sequence validation out of structured frames
        for m in metrics:
            series = pdata[m].to_numpy(dtype=float)
            profile_trajectories[prof][m].append(series)

    grouped_results = {}
    for prof in PAIN_PROFILES:
        grouped_results[prof] = {}
        for m in metrics:
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

    return grouped_results


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
            ax.plot(steps, d["mean"], color=col, lw=2, label="Mean")

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


if __name__ == "__main__":
    # Standard parameters matching previous runs
    N_STEPS = 30
    N_TRAJECTORIES = 100

    print(f"Running execution pipeline: {N_TRAJECTORIES} trajectories across {N_STEPS} steps...")

    # Executes fixed and grouped batch data extraction cleanly
    grouped_res = run_batch_by_profile(
        n_trajectories=N_TRAJECTORIES,
        n_steps=N_STEPS,
        pain_threshold=3.0,
        max_treatments_per_step=3
    )

    plot_batch_by_profile(grouped_res, n_steps=N_STEPS, base_out_path="dashboard_batch_v8")