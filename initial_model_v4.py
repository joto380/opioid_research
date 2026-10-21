import numpy as np
import mesa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import math


"""
Iteration 5: Path-Dependent Pain with Multiple Pain Profiles
-------------------------------------------------------------
Key change from v4: pain is no longer anchored to a fixed base_pain each step.
Instead, pain evolves from the *previous step's pain level*, blended with the
profile's natural drift target.

Pain update formula
-------------------
    new_pain(t) = α * prev_pain(t-1) + (1 - α) * profile_target(t) + noise(t)

Where:
  - α (ALPHA) = 0.2  
  - profile_target(t) is what the profile "wants" pain to be at step t
  - noise is small step-to-step randomness

This means:
  - Treatment that knocks pain down actually shifts the trajectory, not just
    the current value. The patient starts the next step from a genuinely lower
    pain level.
  - Pain still drifts back toward the profile's natural shape, but slowly
    (controlled by α).
  - Each patient is randomly assigned one of four pain profiles at init.

Pain Profiles
-------------
  SINE       : chronic cyclical condition (migraines, arthritis flares)
               target oscillates around base_pain with period ~31 steps

  EXP_DECAY  : acute injury recovering over time
               target starts high, decays exponentially toward a low floor

  LINEAR_INC : degenerative condition
               target rises steadily over the simulation window

  RANDOM_WALK: unpredictable chronic condition
               target itself drifts each step — no underlying trend

Everything else (tiers, addiction, doctor logic, datacollection) is unchanged
from v4.
"""


# ─────────────────────────── Constants ────────────────────────

# Path-dependence strength. 0.0 = fully profile-driven (v4 behaviour),
# 1.0 = pain never returns to profile (random walk only).
ALPHA: float = 0.2

# How much each point of addiction score reduces the effective treatment score
ADDICTION_PENALTY_WEIGHT: float = 0.3

# Score thresholds for tier selection (applied to treatment_score)
HIGH_SCORE_THRESHOLD: float = 6.0
MED_SCORE_THRESHOLD:  float = 3.0

# Addiction score decay per step (metabolism)
ADDICTION_DECAY: float = 0.5

# Tier definitions: (pain reduction, addiction weight)
TIER_EFFECTS = {
    "LOW":    {"reduction": 1.5, "addiction_weight": 0.0},
    "MEDIUM": {"reduction": 3.0, "addiction_weight": 1.0},
    "HIGH":   {"reduction": 5.0, "addiction_weight": 3.0},
}

# Available pain profile types
PAIN_PROFILES = ["SINE", "EXP_DECAY", "LINEAR_INC", "RANDOM_WALK"]


# ─────────────────────────── Pain Profiles ────────────────────

class PainProfile:
    """
    Encapsulates a pain profile for a patient.
    Each profile type implements a different `target(t)` function that
    represents where the underlying condition "wants" to take pain.

    The blending formula in PatientAgent.step() then does:
        new_pain = α * prev_pain + (1-α) * target(t) + noise
    """

    def __init__(self, profile_type: str, base_pain: float, n_steps: int = 50):
        self.profile_type = profile_type
        self.base_pain    = base_pain
        self.n_steps      = n_steps

        # RANDOM_WALK profile needs its own state since it drifts step-to-step
        self._rw_current = base_pain

    def target(self, t: int) -> float:
        """Return the profile's natural pain target at step t."""
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

    def _sine(self, t: int) -> float:
        oscillation = 2.0 * math.sin(t / 5.0)
        return max(0.0, min(10.0, self.base_pain + oscillation))

    def _exp_decay(self, t: int) -> float:
        # Starts at base_pain, decays toward a floor of 1.0
        floor  = 1.0
        decay  = 0.1   # controls how fast pain resolves
        target = floor + (self.base_pain - floor) * math.exp(-decay * t)
        return max(0.0, min(10.0, target))

    def _linear_inc(self, t: int) -> float:
        # Rises by ~0.1 per step, capped at 10
        rate   = 0.1
        target = self.base_pain + rate * t
        return max(0.0, min(10.0, target))

    def _random_walk(self) -> float:
        # Drifts ±0.4 each step, clamped to [0, 10]
        self._rw_current += random.uniform(-0.4, 0.4)
        self._rw_current  = max(0.0, min(10.0, self._rw_current))
        return self._rw_current




# ─────────────────────────── Treatment Tier Logic ─────────────

def compute_treatment_tier(pain: float, addiction_score: float, pain_threshold: float) -> str | None:
    """
    Unchanged from v4.
    Returns tier name ("LOW", "MEDIUM", "HIGH") or None.
    """
    if pain < pain_threshold:
        return None
    treatment_score = pain - (addiction_score * ADDICTION_PENALTY_WEIGHT)
    if treatment_score >= HIGH_SCORE_THRESHOLD:
        return "HIGH"
    elif treatment_score >= MED_SCORE_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


# ─────────────────────────── Agents ───────────────────────────

class PatientAgent(mesa.Agent):
    """
    Patient with path-dependent pain evolution.

    Pain update each step:
        new_pain = α * prev_pain + (1-α) * profile.target(t) + noise

    The patient's pain level now carries memory: being treated last step
    starts this step from a genuinely lower baseline, not just a temporary dip.
    """

    def __init__(self, model, base_pain: float = 5.0):
        super().__init__(model)
        self.base_pain   = base_pain
        self.pain: float = base_pain

        # Randomly assign a pain profile
        profile_type     = random.choice(PAIN_PROFILES)
        self.profile     = PainProfile(profile_type, base_pain)
        self.profile_type = profile_type   # stored for logging / plotting

        self.treated_this_step: bool  = False
        self.current_tier: str | None = None

        # Addiction tracking (unchanged from v4)
        self.medium_treatment_count: int = 0
        self.high_treatment_count: int   = 0
        self.addiction_score: float      = 0.0
        self.downgrades_received: int    = 0

    # ---------- path-dependent pain evolution ----------
    def _evolve_pain(self) -> float:
        t      = self.model.current_step
        target = self.profile.target(t)
        noise  = random.uniform(-0.3, 0.3)

        new_pain = ALPHA * self.pain + (1.0 - ALPHA) * target + noise
        return max(0.0, min(10.0, new_pain))

    # ---------- step ----------
    def step(self):
        self.pain = self._evolve_pain()
        self.treated_this_step = False
        self.current_tier      = None

        # Natural decay of addiction score
        self.addiction_score = max(0.0, self.addiction_score - ADDICTION_DECAY)

    # ---------- receive treatment ----------
    def receive_treatment(self, tier_name: str):
        effects          = TIER_EFFECTS[tier_name]
        reduction        = effects["reduction"]
        addiction_weight = effects["addiction_weight"]

        self.pain = max(0.0, self.pain - reduction)
        self.treated_this_step = True
        self.current_tier      = tier_name

        if tier_name == "MEDIUM":
            self.medium_treatment_count += 1
        elif tier_name == "HIGH":
            self.high_treatment_count += 1

        self.addiction_score += addiction_weight

        print(
            f"  [Step {self.model.current_step}] Patient {self.unique_id} "
            f"({self.profile_type}) | Tier={tier_name} | "
            f"Pain → {self.pain:.2f} | Addiction: {self.addiction_score:.2f}"
        )


class DoctorAgent(mesa.Agent):
    """Unchanged from v4 — composite treatment score, priority queue, downgrade tracking."""

    def __init__(self, model, pain_threshold: float = 3.0, max_treatments_per_step: int = 2):
        super().__init__(model)
        self.pain_threshold          = pain_threshold
        self.max_treatments_per_step = max_treatments_per_step
        self.treatments_given: int   = 0
        self.downgrades_issued: int  = 0

    def step(self):
        treatments_this_step = 0
        patients = list(self.model.agents_by_type[PatientAgent])
        patients.sort(key=lambda p: p.pain, reverse=True)

        for patient in patients:
            if treatments_this_step >= self.max_treatments_per_step:
                print(
                    f"  [Step {self.model.current_step}] Doctor {self.unique_id} "
                    f"reached quota ({self.max_treatments_per_step})."
                )
                break

            raw_tier    = compute_treatment_tier(patient.pain, 0.0, self.pain_threshold)
            chosen_tier = compute_treatment_tier(patient.pain, patient.addiction_score, self.pain_threshold)

            if chosen_tier is None:
                print(
                    f"  [Step {self.model.current_step}] Patient {patient.unique_id} "
                    f"pain={patient.pain:.2f} – below threshold, no treatment."
                )
                continue

            tier_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            if raw_tier is not None and tier_order[chosen_tier] < tier_order[raw_tier]:
                downgrade_msg = (
                    f" ↓ Downgraded {raw_tier}→{chosen_tier} "
                    f"(addiction={patient.addiction_score:.2f}, "
                    f"score={patient.pain - patient.addiction_score * ADDICTION_PENALTY_WEIGHT:.2f})"
                )
                self.downgrades_issued   += 1
                patient.downgrades_received += 1
            else:
                downgrade_msg = ""

            print(
                f"  [Step {self.model.current_step}] Doctor → "
                f"Patient {patient.unique_id} ({patient.profile_type}) | "
                f"pain={patient.pain:.2f} | addiction={patient.addiction_score:.2f} | "
                f"score={patient.pain - patient.addiction_score * ADDICTION_PENALTY_WEIGHT:.2f}"
                + downgrade_msg
            )

            patient.receive_treatment(chosen_tier)
            treatments_this_step    += 1
            self.treatments_given   += 1


# ─────────────────────────── Model ────────────────────────────

class DoctorPatientModel(mesa.Model):
    """Model containing one doctor and multiple patients."""

    def __init__(self, n_patients: int = 3, pain_threshold: float = 3.0, max_treatments_per_step: int = 2):
        super().__init__()
        self.current_step = 0

        for _ in range(n_patients):
            base_pain = random.uniform(3.0, 8.0)
            PatientAgent(model=self, base_pain=base_pain)

        DoctorAgent(model=self, pain_threshold=pain_threshold, max_treatments_per_step=max_treatments_per_step)

        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Pain":             lambda a: a.pain                   if isinstance(a, PatientAgent) else None,
                "Tier":             lambda a: a.current_tier           if isinstance(a, PatientAgent) else None,
                "AddictionScore":   lambda a: a.addiction_score        if isinstance(a, PatientAgent) else None,
                "TreatmentScore":   lambda a: (
                                        a.pain - a.addiction_score * ADDICTION_PENALTY_WEIGHT
                                    )                                  if isinstance(a, PatientAgent) else None,
                "ProfileType":      lambda a: a.profile_type           if isinstance(a, PatientAgent) else None,
                "MediumTreatments": lambda a: a.medium_treatment_count if isinstance(a, PatientAgent) else None,
                "HighTreatments":   lambda a: a.high_treatment_count   if isinstance(a, PatientAgent) else None,
                "Downgrades":       lambda a: a.downgrades_received    if isinstance(a, PatientAgent) else None,
                "Treated":          lambda a: a.treated_this_step      if isinstance(a, PatientAgent) else None,
                "TotalTreatments":  lambda a: a.treatments_given       if isinstance(a, DoctorAgent)  else None,
                "TotalDowngrades":  lambda a: a.downgrades_issued      if isinstance(a, DoctorAgent)  else None,
            }
        )

    def step(self):
        self.datacollector.collect(self)
        self.agents_by_type[PatientAgent].do("step")
        self.agents_by_type[DoctorAgent].do("step")
        self.current_step += 1


# ─────────────────────────── Plotting ─────────────────────────

# Profile-specific colors for the pain trajectory line
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


def plot_results(model, df, n_steps):
    """
    Three-panel figure per patient:
      Top    : pain over time, coloured by treatment tier; profile type in title
      Middle : addiction score over time
      Bottom : treatment score with tier threshold lines
    """
    patient_ids = [a.unique_id for a in model.agents_by_type[PatientAgent]]
    n_patients  = len(patient_ids)
    N_ROWS      = 3

    fig, axes = plt.subplots(
        nrows=N_ROWS, ncols=n_patients,
        figsize=(6 * n_patients, 10),
        sharex=True,
    )
    if n_patients == 1:
        axes = np.array(axes).reshape(N_ROWS, 1)

    fig.suptitle(
        f"Doctor-Patient Simulation v5 — Path-Dependent Pain (α={ALPHA})\n"
        f"treatment_score = pain − addiction × {ADDICTION_PENALTY_WEIGHT}",
        fontsize=13, fontweight="bold"
    )

    for col, pid in enumerate(patient_ids):
        ax_pain   = axes[0, col]
        ax_addict = axes[1, col]
        ax_tscore = axes[2, col]

        pdata = df.xs(pid, level="AgentID") if pid in df.index.get_level_values("AgentID") else None
        if pdata is None or pdata.empty:
            continue

        steps       = pdata.index.get_level_values("Step").tolist()
        pain_vals   = pdata["Pain"].tolist()
        tier_vals   = pdata["Tier"].tolist()
        addict_vals = pdata["AddictionScore"].tolist()
        tscore_vals = pdata["TreatmentScore"].tolist()
        profile     = pdata["ProfileType"].dropna().iloc[0] if not pdata["ProfileType"].dropna().empty else "?"

        profile_color = PROFILE_COLORS.get(profile, "#333333")

        # ── Top: pain coloured by tier, base line in profile color ──
        ax_pain.plot(steps, pain_vals, color=profile_color, linewidth=1.2, alpha=0.4, zorder=1)
        for i in range(len(steps) - 1):
            tier = tier_vals[i]
            ax_pain.plot(
                steps[i:i+2], pain_vals[i:i+2],
                color=TIER_COLORS.get(tier, "#888"),
                linewidth=2.2, zorder=2,
            )
        sc_colors = [TIER_COLORS.get(t, "#888") for t in tier_vals]
        ax_pain.scatter(steps, pain_vals, c=sc_colors, s=40, zorder=5)

        ax_pain.axhspan(0, 3,  color="#4CAF50", alpha=0.06)
        ax_pain.axhspan(3, 6,  color="#FF9800", alpha=0.06)
        ax_pain.axhspan(6, 10, color="#F44336", alpha=0.06)
        ax_pain.axhline(MED_SCORE_THRESHOLD,  color="#FF9800", lw=0.8, ls="--", alpha=0.5)
        ax_pain.axhline(HIGH_SCORE_THRESHOLD, color="#F44336",  lw=0.8, ls="--", alpha=0.5)
        ax_pain.set_ylim(0, 10)
        ax_pain.set_ylabel("Pain (0–10)")
        ax_pain.set_title(f"Patient {pid} — {profile}\nPain & Tier")

        tier_patches = [mpatches.Patch(color=c, label=t) for t, c in TIER_COLORS.items() if t]
        ax_pain.legend(handles=tier_patches, fontsize=7, loc="upper right")

        # ── Middle: addiction score ──
        ax_addict.plot(steps, addict_vals, color="#9C27B0", linewidth=2.0)
        ax_addict.fill_between(steps, addict_vals, alpha=0.15, color="#9C27B0")
        ax_addict.set_ylabel("Addiction score")
        ax_addict.set_title(f"Patient {pid} — Addiction Score")

        # ── Bottom: treatment score ──
        ax_tscore.plot(steps, tscore_vals, color=profile_color, linewidth=2.0, label="Treatment score")
        ax_tscore.fill_between(steps, tscore_vals, alpha=0.12, color=profile_color)
        ax_tscore.axhline(HIGH_SCORE_THRESHOLD, color="#F44336",  lw=1.2, ls="--",
                          label=f"HIGH threshold ({HIGH_SCORE_THRESHOLD})")
        ax_tscore.axhline(MED_SCORE_THRESHOLD,  color="#FF9800", lw=1.2, ls="--",
                          label=f"MED threshold ({MED_SCORE_THRESHOLD})")
        ax_tscore.set_ylabel("Treatment score")
        ax_tscore.set_xlabel("Step")
        ax_tscore.set_title(f"Patient {pid} — Treatment Score")
        ax_tscore.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out_path = "simulation_plots_v5.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()


# ─────────────────────────── Run ──────────────────────────────

if __name__ == "__main__":
    N_STEPS = 30

    model = DoctorPatientModel(
        n_patients=3,
        pain_threshold=3.0,
        max_treatments_per_step=3,
    )

    print("=== Doctor-Patient Simulation v5 (Path-Dependent Pain) ===\n")
    for _ in range(N_STEPS):
        model.step()

    df = model.datacollector.get_agent_vars_dataframe()

    print("\n=== Patient Data (last 15 rows) ===")
    patient_df = df[df["Pain"].notna()]
    print(
        patient_df[["Pain", "Tier", "ProfileType", "AddictionScore",
                     "TreatmentScore", "MediumTreatments", "HighTreatments", "Downgrades"]]
        .tail(15)
        .to_string()
    )

    doctor = next(iter(model.agents_by_type[DoctorAgent]))
    print(f"\nTotal treatments given     : {doctor.treatments_given}")
    print(f"Addiction-driven downgrades: {doctor.downgrades_issued}")

    for a in model.agents_by_type[PatientAgent]:
        print(
            f"Patient {a.unique_id} ({a.profile_type}): "
            f"medium={a.medium_treatment_count}, high={a.high_treatment_count}, "
            f"downgrades={a.downgrades_received}, final_addiction={a.addiction_score:.2f}"
        )

    plot_results(model, df, N_STEPS)