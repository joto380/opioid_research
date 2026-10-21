import numpy as np
import mesa
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import math


"""
The goal of this second iteration of the model is to try and implement the pain ladder that we discussed
and the evolution in terms of treatment that this entails :
- low pain --> low pain reduction, 0 risk
-medium pain --> medium pain reduction, low-med risk
-high pain --> high pain reduction, med-high risk
 
This will also see us implement a "risk"/addiction function which could (for example) count out the number of
med-pain and high-pain treatments.
"""



"""
Iteration 3: Pain Ladder + Addiction/Risk System
-------------------------------------------------
Pain Tiers:
  - LOW    (0–3):  mild analgesic    → small reduction (1.5), zero addiction weight
  - MEDIUM (3–6):  moderate analgesic → medium reduction (3.0), addiction weight +1
  - HIGH   (6–10): strong analgesic   → large reduction (5.0), addiction weight +3

Addiction / Risk System (per patient):
  - medium_treatment_count  : number of medium-tier treatments received
  - high_treatment_count    : number of high-tier treatments received
  - addiction_score         : cumulative score (medium adds 1/step, high adds 3/step),
                              decays slightly each step to model metabolism
  - If addiction_score >= ADDICTION_THRESHOLD the doctor is BLOCKED from giving
    that patient any medium or high treatment until the score drops below the threshold.
"""


# ─────────────────────────── Plotting ─────────────────────────

def plot_results(model, df, n_steps):
    """
    Two-panel figure per patient:
      Top panel   : pain over time, coloured by tier, with addiction-block markers
      Bottom panel: addiction score over time with the block threshold line
    """
    patient_ids = [a.unique_id for a in model.schedule.agents if isinstance(a, PatientAgent)]
    n_patients  = len(patient_ids)

    TIER_COLORS = {"LOW": "#4CAF50", "MEDIUM": "#FF9800", "HIGH": "#F44336"}
    BLOCK_COLOR = "#9C27B0"

    fig, axes = plt.subplots(
        nrows=2, ncols=n_patients,
        figsize=(6 * n_patients, 8),
        sharex=True,
    )
    # Ensure axes is always 2-D
    if n_patients == 1:
        axes = np.array(axes).reshape(2, 1)

    fig.suptitle("Doctor-Patient Simulation — Pain Ladder & Addiction Scores", fontsize=14, fontweight="bold")

    for col, pid in enumerate(patient_ids):
        ax_pain = axes[0, col]
        ax_addict = axes[1, col]

        # ── Slice data for this patient ──
        pdata = df.xs(pid, level="AgentID") if pid in df.index.get_level_values("AgentID") else None
        if pdata is None or pdata.empty:
            continue

        steps = pdata.index.get_level_values("Step").tolist()
        pain_vals   = pdata["Pain"].tolist()
        tier_vals   = pdata["Tier"].tolist()
        addict_vals = pdata["AddictionScore"].tolist()

        # ── Top panel: pain coloured by tier ──
        for i in range(len(steps) - 1):
            tier = tier_vals[i] if tier_vals[i] else "LOW"
            ax_pain.plot(
                steps[i:i+2], pain_vals[i:i+2],
                color=TIER_COLORS.get(tier, "#888"),
                linewidth=2.0,
            )

        # Scatter dots coloured by tier
        scatter_colors = [TIER_COLORS.get(t if t else "LOW", "#888") for t in tier_vals]
        ax_pain.scatter(steps, pain_vals, c=scatter_colors, s=40, zorder=5)

        # Mark steps where treatment was blocked (addiction score >= threshold at that step)
        for i, (s, p, a) in enumerate(zip(steps, pain_vals, addict_vals)):
            if a >= ADDICTION_THRESHOLD:
                ax_pain.axvspan(s - 0.5, s + 0.5, color=BLOCK_COLOR, alpha=0.15)

        # Pain tier shading on background
        ax_pain.axhspan(0, 3,  color="#4CAF50", alpha=0.07)
        ax_pain.axhspan(3, 6,  color="#FF9800", alpha=0.07)
        ax_pain.axhspan(6, 10, color="#F44336", alpha=0.07)

        ax_pain.set_ylim(0, 10)
        ax_pain.set_ylabel("Pain level (0–10)")
        ax_pain.set_title(f"Patient {pid} — Pain over Time")
        ax_pain.axhline(3, color="#FF9800", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_pain.axhline(6, color="#F44336", linewidth=0.8, linestyle="--", alpha=0.5)

        # Legend for pain panel
        patches = [mpatches.Patch(color=c, label=t) for t, c in TIER_COLORS.items()]
        block_patch = mpatches.Patch(color=BLOCK_COLOR, alpha=0.3, label="Addiction block zone")
        ax_pain.legend(handles=patches + [block_patch], fontsize=8, loc="upper right")

        # ── Bottom panel: addiction score ──
        ax_addict.plot(steps, addict_vals, color="#9C27B0", linewidth=2.0, label="Addiction score")
        ax_addict.fill_between(steps, addict_vals, alpha=0.15, color="#9C27B0")
        ax_addict.axhline(ADDICTION_THRESHOLD, color="red", linewidth=1.2,
                          linestyle="--", label=f"Block threshold ({ADDICTION_THRESHOLD})")

        ax_addict.set_ylim(bottom=0)
        ax_addict.set_ylabel("Addiction score")
        ax_addict.set_xlabel("Step")
        ax_addict.set_title(f"Patient {pid} — Addiction Score")
        ax_addict.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = "simulation_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()

# ─────────────────────────── Constants ────────────────────────

ADDICTION_THRESHOLD = 10.0   # doctor blocked above this score
ADDICTION_DECAY     = 0.5    # score naturally drops by this amount each step

# Tier definitions: (label, pain_floor, reduction, addiction_weight)
TIERS = [
    ("LOW",    0.0,  3.0, 1.5, 0.0),   # (name, min_pain, max_pain, reduction, addict_weight)
    ("MEDIUM", 3.0,  6.0, 3.0, 1.0),
    ("HIGH",   6.0, 10.0, 5.0, 3.0),
]

def get_tier(pain: float):
    """Return (tier_name, reduction, addiction_weight) for a given pain level."""
    for name, lo, hi, reduction, weight in TIERS:
        if lo <= pain < hi or (name == "HIGH" and pain >= hi):
            return name, reduction, weight
    return "LOW", 1.5, 0.0   # fallback


# ─────────────────────────── Agents ───────────────────────────

class PatientAgent(mesa.Agent):
    """A patient whose pain oscillates and can be reduced by tiered treatment."""

    def __init__(self, unique_id, model, base_pain: float = 5.0):
        super().__init__(unique_id, model)
        self.base_pain = base_pain
        self.pain: float = base_pain
        self.treated_this_step: bool = False
        self.current_tier: str = "LOW"

        # ── Addiction / risk tracking ──
        self.medium_treatment_count: int   = 0
        self.high_treatment_count: int     = 0
        self.addiction_score: float        = 0.0

    # ---------- pain evolution ----------
    def _natural_pain_evolution(self) -> float:
        t = self.model.schedule.steps
        oscillation = 2.0 * math.sin(t / 5.0)
        noise = random.uniform(-0.5, 0.5)
        new_pain = self.base_pain + oscillation + noise
        return max(0.0, min(10.0, new_pain))

    # ---------- step ----------
    def step(self):
        self.pain = self._natural_pain_evolution()
        self.treated_this_step = False

        # Natural decay of addiction score each step
        self.addiction_score = max(0.0, self.addiction_score - ADDICTION_DECAY)

    # ---------- treatment ----------
    def receive_treatment(self, tier_name: str, reduction: float, addiction_weight: float):
        self.pain = max(0.0, self.pain - reduction)
        self.treated_this_step = True
        self.current_tier = tier_name

        # Update addiction counters
        if tier_name == "MEDIUM":
            self.medium_treatment_count += 1
            self.addiction_score += addiction_weight
        elif tier_name == "HIGH":
            self.high_treatment_count += 1
            self.addiction_score += addiction_weight

        print(
            f"  [Step {self.model.schedule.steps}] Patient {self.unique_id} | "
            f"Tier={tier_name} | Pain → {self.pain:.2f} | "
            f"Addiction score: {self.addiction_score:.2f}"
        )


class DoctorAgent(mesa.Agent):
    """
    Doctor who:
      1. Reads patient pain level → determines appropriate tier
      2. Checks patient addiction score — if >= ADDICTION_THRESHOLD, only LOW
         treatment is permitted (or no treatment if pain is already low)
      3. Treats up to max_treatments_per_step patients per step (highest pain first)
    """

    def __init__(
        self,
        unique_id,
        model,
        pain_threshold: float = 3.0,        # minimum pain to trigger any treatment
        max_treatments_per_step: int = 2,
    ):
        super().__init__(unique_id, model)
        self.pain_threshold = pain_threshold
        self.max_treatments_per_step = max_treatments_per_step
        self.treatments_given: int = 0
        self.blocked_treatments: int = 0     # times a stronger treatment was blocked

    def step(self):
        treatments_this_step = 0

        patients = [a for a in self.model.schedule.agents if isinstance(a, PatientAgent)]
        patients.sort(key=lambda p: p.pain, reverse=True)

        for patient in patients:
            if treatments_this_step >= self.max_treatments_per_step:
                print(
                    f"  [Step {self.model.schedule.steps}] Doctor {self.unique_id} "
                    f"reached quota ({self.max_treatments_per_step})."
                )
                break

            if patient.pain < self.pain_threshold:
                print(
                    f"  [Step {self.model.schedule.steps}] Patient {patient.unique_id} "
                    f"pain={patient.pain:.2f} below threshold – no treatment."
                )
                continue

            # Determine appropriate tier from pain level
            tier_name, reduction, addiction_weight = get_tier(patient.pain)

            # ── Addiction gate ──
            if patient.addiction_score >= ADDICTION_THRESHOLD:
                if tier_name in ("MEDIUM", "HIGH"):
                    print(
                        f"  [Step {self.model.schedule.steps}] ⚠ Patient {patient.unique_id} "
                        f"BLOCKED from {tier_name} treatment "
                        f"(addiction score={patient.addiction_score:.2f}). "
                        f"Downgrading to LOW."
                    )
                    self.blocked_treatments += 1
                    tier_name, reduction, addiction_weight = "LOW", 1.5, 0.0

            patient.receive_treatment(tier_name, reduction, addiction_weight)
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
        self.schedule = mesa.time.RandomActivation(self)

        for i in range(n_patients):
            base_pain = random.uniform(3.0, 8.0)
            patient = PatientAgent(unique_id=i, model=self, base_pain=base_pain)
            self.schedule.add(patient)

        doctor = DoctorAgent(
            unique_id=n_patients,
            model=self,
            pain_threshold=pain_threshold,
            max_treatments_per_step=max_treatments_per_step,
        )
        self.schedule.add(doctor)

        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Pain":                  lambda a: a.pain if isinstance(a, PatientAgent) else None,
                "Tier":                  lambda a: a.current_tier if isinstance(a, PatientAgent) else None,
                "AddictionScore":        lambda a: a.addiction_score if isinstance(a, PatientAgent) else None,
                "MediumTreatments":      lambda a: a.medium_treatment_count if isinstance(a, PatientAgent) else None,
                "HighTreatments":        lambda a: a.high_treatment_count if isinstance(a, PatientAgent) else None,
                "Treated":               lambda a: a.treated_this_step if isinstance(a, PatientAgent) else None,
                "TotalTreatmentsGiven":  lambda a: a.treatments_given if isinstance(a, DoctorAgent) else None,
                "BlockedTreatments":     lambda a: a.blocked_treatments if isinstance(a, DoctorAgent) else None,
            }
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


# ─────────────────────────── Run ──────────────────────────────

if __name__ == "__main__":
    N_STEPS = 30

    model = DoctorPatientModel(
        n_patients=3,
        pain_threshold=3.0,
        max_treatments_per_step=2,
    )

    print("=== Doctor-Patient Simulation (Pain Ladder) ===\n")
    for _ in range(N_STEPS):
        model.step()

    df = model.datacollector.get_agent_vars_dataframe()

    print("\n=== Agent Data (last 10 steps) ===")
    patient_df = df[df["Pain"].notna()]
    print(patient_df[["Pain", "Tier", "AddictionScore", "MediumTreatments", "HighTreatments"]].tail(15).to_string())

    doctor = next(a for a in model.schedule.agents if isinstance(a, DoctorAgent))
    print(f"\nTotal treatments given:   {doctor.treatments_given}")
    print(f"Blocked (addiction gate): {doctor.blocked_treatments}")

    for a in model.schedule.agents:
        if isinstance(a, PatientAgent):
            print(
                f"Patient {a.unique_id}: medium_treatments={a.medium_treatment_count}, "
                f"high_treatments={a.high_treatment_count}, "
                f"final_addiction_score={a.addiction_score:.2f}"
            )

    plot_results(model, df, N_STEPS)


