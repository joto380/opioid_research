import numpy as np
import mesa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import math
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
    """
    Fixed Assumption: Shifted from fragile linear decay to standard pharmacology
    Sigmoidal Hill/Emax structure. Efficacy loss asymptotic profile.
    """

    def decay_step(self, current_tolerance: float) -> float:
        # Multiplicative/Concentration-dependent passive clearance
        return max(0.0, current_tolerance * (1.0 - TOLERANCE_DECAY_RATE))

    def gain_from_dose(self, tier_name: str) -> float:
        if tier_name == "MEDIUM": return TOLERANCE_GAIN_MEDIUM
        if tier_name == "HIGH":   return TOLERANCE_GAIN_HIGH
        return 0.0

    def apply_to_reduction(self, base_reduction: float, tolerance: float) -> float:
        # Sigmoidal Multiplier: Efficacy = 1 / (1 + (Tol / Tol_50)^2)
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

        # Explicit Rolling Trackers for Moving Windows
        self.pain_history: List[float] = [base_pain]
        self.addiction_history: List[float] = [0.0]
        self.profile_targets: List[float] = []

    @property
    def rolling_pain_auc(self) -> float:
        # Computes Trapezoidal AUC over the designated sliding memory window
        history = self.pain_history[-AUC_MEM_WINDOW:]
        if len(history) < 2: return 0.0

        # NumPy 2.0+ / Python 3.13 compatibility fix
        if hasattr(np, 'trapezoid'):
            return float(np.trapezoid(history))
        return float(np.trapz(history))

    @property
    def rolling_addiction_auc(self) -> float:
        history = self.addiction_history[-AUC_MEM_WINDOW:]
        if len(history) < 2: return 0.0

        # NumPy 2.0+ / Python 3.13 compatibility fix
        if hasattr(np, 'trapezoid'):
            return float(np.trapezoid(history))
        return float(np.trapz(history))

    def pre_treatment_update(self):
        """Phase 1 of Simultaneous Activation: Evolve baseline pain & decay scores"""
        t = self.model.current_step
        target = self.profile.target(t)
        self.profile_targets.append(target)

        # Pain Evolution
        noise = random.uniform(-0.3, 0.3)
        self.pain = max(0.0, min(10.0, ALPHA * self.pain + (1.0 - ALPHA) * target + noise))

        # Natural Degradations
        self.steps_since_last_dose += 1
        self.addiction_score = self.addiction_dynamic.decay(self.addiction_score, t)
        self.tolerance = self.tolerance_dynamic.decay_step(self.tolerance)
        self.craving = self.craving_dynamic.decay_step(self.craving, self.pain)

        # Reset staging variables
        self.treated_this_step = False
        self.current_tier = None

    def receive_treatment(self, tier_name: str):
        effects = TIER_EFFECTS[tier_name]
        reduction = effects["reduction"]
        addiction_weight = effects["addiction_weight"]

        # Local window evaluation instead of globally diluted step counts
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
        """Phase 3: Finalize historical logs & resolve standard post-treatment metrics"""
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
        """Phase 2: Doctor processes patients who have updated baseline pain values"""
        treatments_this_step = 0
        patients = list(self.model.agents_by_type[PatientAgent])
        patients.sort(key=lambda p: p.pain, reverse=True)

        for patient in patients:
            if treatments_this_step >= self.max_treatments_per_step:
                break

            # Fixed Blindspot: Doctor evaluates systemic clinical markers including tolerance impact
            if patient.pain < self.pain_threshold:
                continue

            # Clinical Decision Optimization Engine
            effective_pain_score = patient.pain - (patient.addiction_score * ADDICTION_PENALTY_WEIGHT)

            # Sub-clinical heuristic penalty if tolerance has completely ruined efficacy
            if patient.tolerance > 6.0:
                effective_pain_score -= (patient.tolerance * 0.2)

            if effective_pain_score >= HIGH_SCORE_THRESHOLD:
                chosen_tier = "HIGH"
            elif effective_pain_score >= MED_SCORE_THRESHOLD:
                chosen_tier = "MEDIUM"
            else:
                chosen_tier = "LOW"

            # Pure symptom tier for baseline evaluation
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
    def __init__(self, n_patients: int = 3, pain_threshold: float = 3.0, max_treatments_per_step: int = 2,
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
                "PainAUC": lambda a: a.rolling_pain_auc if isinstance(a, PatientAgent) else None,
                "AddictionAUC": lambda a: a.rolling_addiction_auc if isinstance(a, PatientAgent) else None,
            }
        )

    def step(self):
        patients = self.agents_by_type[PatientAgent]
        doctors = self.agents_by_type[DoctorAgent]

        # Execution Pipeline Fix (Three-Stage Simultaneous Staging)
        patients.do("pre_treatment_update")
        doctors.do("step")
        patients.do("post_treatment_cleanup")

        self.datacollector.collect(self)
        self.current_step += 1


# ─────────────────────────── OPTIMIZED PROCESSING & PLOTS ────────────────

def run_batch_optimized(n_trajectories: int = 50, n_steps: int = 30) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Performance Optimization: Avoids inner-loop multi-index dataframe extraction slicing.
    Extracts underlying raw Python arrays instantly out of execution frames.
    """
    metrics = ["Pain", "AddictionScore", "Tolerance", "Craving", "Functionality"]
    raw_trajectories = {m: [] for m in metrics}

    for _ in range(n_trajectories):
        model = DoctorPatientModel(n_patients=1, max_treatments_per_step=1)
        for _ in range(n_steps):
            model.step()

        # Unpack raw instances straight out of system lists to jump memory allocations
        patient = list(model.agents_by_type[PatientAgent])[0]

        df = model.datacollector.get_agent_vars_dataframe().xs(patient.unique_id, level="AgentID")
        for m in metrics:
            raw_trajectories[m].append(df[m].to_numpy(dtype=float))

    results = {}
    for m in metrics:
        arr = np.array(raw_trajectories[m])
        results[m] = {
            "mean": np.nanmean(arr, axis=0),
            "p5": np.nanpercentile(arr, 5, axis=0),
            "p25": np.nanpercentile(arr, 25, axis=0),
            "p75": np.nanpercentile(arr, 75, axis=0),
            "p95": np.nanpercentile(arr, 95, axis=0),
            "all": arr
        }
    return results


def plot_batch_dashboard(results: Dict[str, Any], n_steps: int, n_trajectories: int,
                         out_path: str = "optimized_batch.png"):
    metrics = ["Pain", "AddictionScore", "Tolerance", "Craving", "Functionality"]
    titles = {m: f"{m} (Monte Carlo)" for m in metrics}
    colors = {"Pain": "#E53935", "AddictionScore": "#FB8C00", "Tolerance": "#5C6BC0", "Craving": "#EC407A",
              "Functionality": "#43A047"}

    steps = np.arange(n_steps)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Optimized Simulation Dashboard — {n_trajectories} Trajectories", fontsize=12, fontweight='bold')

    for ax, m in zip(axes, metrics):
        d = results[m]
        col = colors[m]
        ax.fill_between(steps, d["p5"], d["p95"], color=col, alpha=0.1)
        ax.fill_between(steps, d["p25"], d["p75"], color=col, alpha=0.25)
        ax.plot(steps, d["mean"], color=col, lw=2, label="Mean")
        ax.set_ylabel(titles[m], fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Steps")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"File pipeline rendered and successfully exported to -> {out_path}")


if __name__ == "__main__":
    # Execute highly accelerated optimized simulation pipeline
    res = run_batch_optimized(n_trajectories=100, n_steps=30)
    plot_batch_dashboard(res, n_steps=30, n_trajectories=100)