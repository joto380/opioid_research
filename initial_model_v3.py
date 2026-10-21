import numpy as np
import mesa
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import math


"""
Iteration 4: Pain Ladder + Composite Treatment Score
-----------------------------------------------------
The addiction score is no longer a hard cap / gate.
Instead, the doctor computes a *treatment score* that blends pain level
and accumulated addiction risk, then maps that score to a tier.

Treatment Score formula
-----------------------
    treatment_score = pain_level - (addiction_score * ADDICTION_PENALTY_WEIGHT)

Tier mapping (on treatment_score, not raw pain):
  treatment_score >= HIGH_SCORE_THRESHOLD  → HIGH   treatment
  treatment_score >= MED_SCORE_THRESHOLD   → MEDIUM treatment
  treatment_score >= pain_threshold        → LOW    treatment
  otherwise                               → no treatment

Effect:
  - A patient with high pain but low addiction gets the expected tier.
  - A patient with high pain AND high addiction may be downgraded one tier
    because the penalty reduces the effective score.
  - Unlike the hard gate, very high pain can still justify a strong treatment
    even with some addiction risk — the doctor just needs a higher raw pain
    level to cross the same tier boundary.

Pain still evolves naturally (sinusoidal + noise) and addiction still
decays each step to model metabolism / tolerance reset.
"""


# ─────────────────────────── Constants ────────────────────────

# How much each point of addiction score reduces the effective treatment score
ADDICTION_PENALTY_WEIGHT: float = 0.3

# Score thresholds that determine which tier is selected
# (applied to treatment_score, not raw pain)
HIGH_SCORE_THRESHOLD: float = 6.0    # treatment_score >= this → HIGH
MED_SCORE_THRESHOLD:  float = 3.0    # treatment_score >= this → MEDIUM
# below MED_SCORE_THRESHOLD (but above pain_threshold) → LOW

# Addiction score still accumulates and decays
ADDICTION_DECAY: float = 0.5   # score drops by this amount each step (metabolism)

# Tier definitions: (label, reduction, addiction_weight)
# Reduction = how much pain is reduced when this tier is applied
TIER_EFFECTS = {
    "LOW":    {"reduction": 1.5, "addiction_weight": 0.0},
    "MEDIUM": {"reduction": 3.0, "addiction_weight": 1.0},
    "HIGH":   {"reduction": 5.0, "addiction_weight": 3.0},
}


def compute_treatment_tier(pain: float, addiction_score: float, pain_threshold: float) -> str | None:
    """
    Compute which treatment tier the doctor should use, given:
      - pain           : current patient pain level (0–10)
      - addiction_score: accumulated addiction/risk score for this patient
      - pain_threshold : minimum raw pain level to trigger any treatment

    Returns the tier name ("LOW", "MEDIUM", "HIGH") or None if no treatment.

    The *treatment score* penalises the raw pain level by the addiction risk,
    nudging the doctor toward a lower tier when the patient has built up risk.
    The penalty is *soft*: a patient in severe pain (e.g. 9/10) can still
    receive HIGH treatment even with moderate addiction risk, whereas a patient
    at borderline HIGH pain (e.g. 6.5) may be downgraded to MEDIUM.
    """
    # No treatment below the raw pain threshold
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
    """A patient whose pain oscillates naturally and can be reduced by tiered treatment."""

    def __init__(self, model, base_pain: float = 5.0):
        super().__init__(model)
        self.base_pain = base_pain
        self.pain: float = base_pain
        self.treated_this_step: bool = False
        self.current_tier: str | None = None

        # ── Addiction / risk tracking ──
        self.medium_treatment_count: int = 0
        self.high_treatment_count: int   = 0
        self.addiction_score: float      = 0.0

        # Track how often the doctor downgraded treatment due to addiction risk
        self.downgrades_received: int = 0

    # ---------- natural pain evolution ----------
    def _natural_pain_evolution(self) -> float:
        t = self.model.current_step
        oscillation = 2.0 * math.sin(t / 5.0)
        noise = random.uniform(-0.5, 0.5)
        new_pain = self.base_pain + oscillation + noise
        return max(0.0, min(10.0, new_pain))

    # ---------- step ----------
    def step(self):
        self.pain = self._natural_pain_evolution()
        self.treated_this_step = False
        self.current_tier = None

        # Natural decay of addiction score each step
        self.addiction_score = max(0.0, self.addiction_score - ADDICTION_DECAY)

    # ---------- treatment ----------
    def receive_treatment(self, tier_name: str):
        effects = TIER_EFFECTS[tier_name]
        reduction        = effects["reduction"]
        addiction_weight = effects["addiction_weight"]

        self.pain = max(0.0, self.pain - reduction)
        self.treated_this_step = True
        self.current_tier = tier_name

        # Update addiction counters
        if tier_name == "MEDIUM":
            self.medium_treatment_count += 1
        elif tier_name == "HIGH":
            self.high_treatment_count += 1

        self.addiction_score += addiction_weight

        print(
            f"  [Step {self.model.current_step}] Patient {self.unique_id} | "
            f"Tier={tier_name} | Pain → {self.pain:.2f} | "
            f"Addiction score: {self.addiction_score:.2f}"
        )


class DoctorAgent(mesa.Agent):
    """
    Doctor who:
      1. Computes a treatment_score = pain - (addiction_score × penalty_weight)
      2. Maps that score to a tier — naturally downgrading when addiction risk is high
      3. Treats up to max_treatments_per_step patients per step (highest pain first)

    No hard addiction gate: the composite score handles it continuously.
    """

    def __init__(
        self,
        model,
        pain_threshold: float = 3.0,
        max_treatments_per_step: int = 2,
    ):
        super().__init__(model)
        self.pain_threshold          = pain_threshold
        self.max_treatments_per_step = max_treatments_per_step
        self.treatments_given: int   = 0
        self.downgrades_issued: int  = 0   # times addiction risk caused a lower tier

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

            # What tier does raw pain suggest?
            raw_tier = compute_treatment_tier(patient.pain, 0.0, self.pain_threshold)

            # What tier does the composite score suggest?
            chosen_tier = compute_treatment_tier(
                patient.pain, patient.addiction_score, self.pain_threshold
            )

            if chosen_tier is None:
                print(
                    f"  [Step {self.model.current_step}] Patient {patient.unique_id} "
                    f"pain={patient.pain:.2f} (score below threshold) – no treatment."
                )
                continue

            # Detect and log a downgrade caused by addiction risk
            tier_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            if raw_tier is not None and tier_order[chosen_tier] < tier_order[raw_tier]:
                downgrade_msg = (
                    f" ↓ Downgraded {raw_tier}→{chosen_tier} "
                    f"(addiction={patient.addiction_score:.2f}, "
                    f"treatment_score={patient.pain - patient.addiction_score * ADDICTION_PENALTY_WEIGHT:.2f})"
                )
                self.downgrades_issued += 1
                patient.downgrades_received += 1
            else:
                downgrade_msg = ""

            print(
                f"  [Step {self.model.current_step}] Doctor {self.unique_id} → "
                f"Patient {patient.unique_id} | pain={patient.pain:.2f} | "
                f"addiction={patient.addiction_score:.2f} | "
                f"treatment_score={patient.pain - patient.addiction_score * ADDICTION_PENALTY_WEIGHT:.2f}"
                + downgrade_msg
            )

            patient.receive_treatment(chosen_tier)
            treatments_this_step += 1
            self.treatments_given += 1


# ─────────────────────────── Model ────────────────────────────

class DoctorPatientModel(mesa.Model):
    """Model containing one doctor and multiple patients."""

    def __init__(
        self,
        n_patients: int = 3,
        pain_threshold: float = 3.0,
        max_treatments_per_step: int = 2,
    ):
        super().__init__()
        self.current_step = 0

        for i in range(n_patients):
            base_pain = random.uniform(3.0, 8.0)
            PatientAgent(model=self, base_pain=base_pain)

        DoctorAgent(
            model=self,
            pain_threshold=pain_threshold,
            max_treatments_per_step=max_treatments_per_step,
        )

        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Pain":             lambda a: a.pain                  if isinstance(a, PatientAgent) else None,
                "Tier":             lambda a: a.current_tier          if isinstance(a, PatientAgent) else None,
                "AddictionScore":   lambda a: a.addiction_score       if isinstance(a, PatientAgent) else None,
                "TreatmentScore":   lambda a: (
                                        a.pain - a.addiction_score * ADDICTION_PENALTY_WEIGHT
                                    )                                 if isinstance(a, PatientAgent) else None,
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
        # Patients step first, then doctor
        self.agents_by_type[PatientAgent].do("step")
        self.agents_by_type[DoctorAgent].do("step")
        self.current_step += 1


# ─────────────────────────── Plotting ─────────────────────────

def plot_results(model, df, n_steps):
    """
    Three-panel figure per patient:
      Top    : pain over time, coloured by tier chosen
      Middle : addiction score over time
      Bottom : treatment score (pain − addiction penalty) with tier thresholds
    """
    patient_ids = [a.unique_id for a in model.agents_by_type[PatientAgent]]
    n_patients  = len(patient_ids)

    TIER_COLORS = {"LOW": "#4CAF50", "MEDIUM": "#FF9800", "HIGH": "#F44336", None: "#AAAAAA"}
    N_ROWS = 3

    fig, axes = plt.subplots(
        nrows=N_ROWS, ncols=n_patients,
        figsize=(6 * n_patients, 10),
        sharex=True,
    )
    if n_patients == 1:
        axes = np.array(axes).reshape(N_ROWS, 1)

    fig.suptitle(
        "Doctor-Patient Simulation — Composite Treatment Score\n"
        f"(treatment_score = pain − addiction × {ADDICTION_PENALTY_WEIGHT})",
        fontsize=13, fontweight="bold"
    )

    for col, pid in enumerate(patient_ids):
        ax_pain    = axes[0, col]
        ax_addict  = axes[1, col]
        ax_tscore  = axes[2, col]

        pdata = df.xs(pid, level="AgentID") if pid in df.index.get_level_values("AgentID") else None
        if pdata is None or pdata.empty:
            continue

        steps        = pdata.index.get_level_values("Step").tolist()
        pain_vals    = pdata["Pain"].tolist()
        tier_vals    = pdata["Tier"].tolist()
        addict_vals  = pdata["AddictionScore"].tolist()
        tscore_vals  = pdata["TreatmentScore"].tolist()

        # ── Top: pain coloured by tier ──
        for i in range(len(steps) - 1):
            tier = tier_vals[i]
            ax_pain.plot(
                steps[i:i+2], pain_vals[i:i+2],
                color=TIER_COLORS.get(tier, "#888"),
                linewidth=2.0,
            )
        sc_colors = [TIER_COLORS.get(t, "#888") for t in tier_vals]
        ax_pain.scatter(steps, pain_vals, c=sc_colors, s=40, zorder=5)

        ax_pain.axhspan(0, 3,  color="#4CAF50", alpha=0.07)
        ax_pain.axhspan(3, 6,  color="#FF9800", alpha=0.07)
        ax_pain.axhspan(6, 10, color="#F44336", alpha=0.07)
        ax_pain.axhline(MED_SCORE_THRESHOLD,  color="#FF9800", lw=0.8, ls="--", alpha=0.5)
        ax_pain.axhline(HIGH_SCORE_THRESHOLD, color="#F44336",  lw=0.8, ls="--", alpha=0.5)
        ax_pain.set_ylim(0, 10)
        ax_pain.set_ylabel("Pain (0–10)")
        ax_pain.set_title(f"Patient {pid} — Pain & Tier")
        patches = [mpatches.Patch(color=c, label=t) for t, c in TIER_COLORS.items() if t]
        ax_pain.legend(handles=patches, fontsize=7, loc="upper right")

        # ── Middle: addiction score ──
        ax_addict.plot(steps, addict_vals, color="#9C27B0", linewidth=2.0)
        ax_addict.fill_between(steps, addict_vals, alpha=0.15, color="#9C27B0")
        ax_addict.set_ylabel("Addiction score")
        ax_addict.set_title(f"Patient {pid} — Addiction Score")

        # ── Bottom: treatment score with tier boundary lines ──
        ax_tscore.plot(steps, tscore_vals, color="#1565C0", linewidth=2.0, label="Treatment score")
        ax_tscore.fill_between(steps, tscore_vals, alpha=0.12, color="#1565C0")
        ax_tscore.axhline(HIGH_SCORE_THRESHOLD, color="#F44336",  lw=1.2, ls="--",
                          label=f"HIGH threshold ({HIGH_SCORE_THRESHOLD})")
        ax_tscore.axhline(MED_SCORE_THRESHOLD,  color="#FF9800", lw=1.2, ls="--",
                          label=f"MED threshold ({MED_SCORE_THRESHOLD})")
        ax_tscore.set_ylabel("Treatment score")
        ax_tscore.set_xlabel("Step")
        ax_tscore.set_title(f"Patient {pid} — Treatment Score")
        ax_tscore.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out_path = "simulation_plots_v4.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()


# ─────────────────────────── Run ──────────────────────────────

if __name__ == "__main__":
    N_STEPS = 30

    model = DoctorPatientModel(
        n_patients=3,
        pain_threshold=3.0,
        max_treatments_per_step=2,
    )

    print("=== Doctor-Patient Simulation (Composite Treatment Score) ===\n")
    for _ in range(N_STEPS):
        model.step()

    df = model.datacollector.get_agent_vars_dataframe()

    print("\n=== Patient Data (last 15 rows) ===")
    patient_df = df[df["Pain"].notna()]
    print(
        patient_df[["Pain", "Tier", "AddictionScore", "TreatmentScore",
                     "MediumTreatments", "HighTreatments", "Downgrades"]]
        .tail(15)
        .to_string()
    )

    doctor = next(iter(model.agents_by_type[DoctorAgent]))
    print(f"\nTotal treatments given : {doctor.treatments_given}")
    print(f"Addiction-driven downgrades: {doctor.downgrades_issued}")

    for a in model.agents_by_type[PatientAgent]:
        print(
            f"Patient {a.unique_id}: "
            f"medium={a.medium_treatment_count}, "
            f"high={a.high_treatment_count}, "
            f"downgrades={a.downgrades_received}, "
            f"final_addiction={a.addiction_score:.2f}"
        )

    plot_results(model, df, N_STEPS)