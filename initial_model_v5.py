import numpy as np
import mesa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import math


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
    """
    Governs how addiction score decays each step.

    CONSTANT : score -= base_rate  (flat clearance)
    LINEAR   : score -= base_rate * (1 + decay_rate * t)  (improving clearance)
    EXP      : score  = score * exp(-lambda_ * steps_since_dose)
               Pharmacokinetic interpretation: the score represents the peak
               addiction load from the last addictive dose; it decays
               exponentially based on how many steps have elapsed since that
               dose was received.  Receiving a new medium/high dose resets
               steps_since_dose to 0, reanchoring the decay curve to the new
               (higher) score — so back-to-back dosing sustains addiction
               rather than letting it clear between doses.
    """

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
            # Absolute decay from the last dose: score * e^(-λ * k)
            # where k = steps elapsed since the most recent addictive treatment.
            # This correctly sustains high scores during frequent dosing and
            # only clears once the patient goes untreated for several steps.
            return current_score * math.exp(-self.lambda_ * steps_since_dose)


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


# ─────────────────────────── Agents ───────────────────────────

class PatientAgent(mesa.Agent):
    """
    Patient with path-dependent pain, pluggable addiction decay, and
    a running Area Under the Pain Curve (AUC) computed via the trapezoid rule.

    pain_auc accumulates each step as:
        pain_auc += 0.5 * (pain_at_previous_step + pain_at_current_step) * Δt
    where Δt = 1 step.  It is a read-only property for DoctorAgent to consult.

    profile_targets records the raw pain-profile target value at each step,
    before the EWMA smoothing and noise are applied.  Used by plot_results to
    draw the "ideal" profile as a dashed overlay on the actual pain trace.
    """

    def __init__(
        self,
        model,
        base_pain:            float = 5.0,
        addiction_decay_mode: str   = "CONSTANT",
    ):
        super().__init__(model)
        self.base_pain   = base_pain
        self.pain: float = base_pain

        profile_type      = random.choice(PAIN_PROFILES)
        self.profile      = PainProfile(profile_type, base_pain)
        self.profile_type = profile_type

        self.addiction_dynamic = AddictionDynamic(mode=addiction_decay_mode)

        self.treated_this_step: bool  = False
        self.current_tier: str | None = None

        self.medium_treatment_count: int = 0
        self.high_treatment_count:   int = 0
        self.addiction_score:      float = 0.0
        self.downgrades_received:    int = 0

        # ── Dose-gap counter (used by EXP addiction decay) ────────────────
        # Reset to 0 whenever a medium or high treatment is received.
        # Incremented every step so the EXP decay knows how long it has been
        # since the last addictive dose was administered.
        self.steps_since_last_dose: int = 0
        # ─────────────────────────────────────────────────────────────────

        # ── AUC tracking ─────────────────────────────────────────────────
        self._prev_pain: float = base_pain
        self.pain_auc:   float = 0.0
        # ─────────────────────────────────────────────────────────────────

        # ── Profile-target history (for plotting overlay) ─────────────────
        # Populated in step() before the smoothed pain is computed so that
        # RANDOM_WALK advances only once per step.
        self.profile_targets: list[float] = []
        # ─────────────────────────────────────────────────────────────────

    def _evolve_pain(self, target: float) -> float:
        """
        Compute next pain value given a pre-evaluated profile target.
        Accepting target as a parameter avoids calling profile.target() twice,
        which would advance RANDOM_WALK an extra step.
        """
        noise = random.uniform(-0.3, 0.3)
        return max(0.0, min(10.0, ALPHA * self.pain + (1.0 - ALPHA) * target + noise))

    def step(self):
        t      = self.model.current_step
        target = self.profile.target(t)          # evaluate once …
        self.profile_targets.append(target)      # … record for plotting …
        self.pain = self._evolve_pain(target)    # … and reuse in EWMA

        # Accumulate AUC via trapezoid rule before resetting _prev_pain
        self.pain_auc  += 0.5 * (self._prev_pain + self.pain)   # Δt = 1
        self._prev_pain = self.pain

        self.treated_this_step = False
        self.current_tier      = None

        self.steps_since_last_dose += 1
        self.addiction_score = self.addiction_dynamic.decay(
            self.addiction_score,
            self.model.current_step,
            self.steps_since_last_dose,
        )

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

        # Reset dose-gap counter so EXP decay restarts from this new dose
        if tier_name in ("MEDIUM", "HIGH"):
            self.steps_since_last_dose = 0

        print(
            f"  [Step {self.model.current_step}] Patient {self.unique_id} "
            f"({self.profile_type}, {self.addiction_dynamic.mode}) | "
            f"Tier={tier_name} | Pain→{self.pain:.2f} | "
            f"Addiction: {self.addiction_score:.2f} | AUC: {self.pain_auc:.2f}"
        )


class DoctorAgent(mesa.Agent):
    """
    Doctor agent.  patient.pain_auc is available for future triage logic
    but is not used in the current treatment decision.
    """

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
                print(
                    f"  [Step {self.model.current_step}] Doctor {self.unique_id} "
                    f"reached quota ({self.max_treatments_per_step})."
                )
                break

            raw_tier    = compute_treatment_tier(patient.pain, 0.0, self.pain_threshold)
            chosen_tier = compute_treatment_tier(
                patient.pain, patient.addiction_score, self.pain_threshold
            )

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
                    f"(addiction={patient.addiction_score:.2f})"
                )
                self.downgrades_issued      += 1
                patient.downgrades_received += 1
            else:
                downgrade_msg = ""

            print(
                f"  [Step {self.model.current_step}] Doctor → "
                f"Patient {patient.unique_id} ({patient.profile_type}) | "
                f"pain={patient.pain:.2f} | addiction={patient.addiction_score:.2f} | "
                f"auc={patient.pain_auc:.1f}"
                + downgrade_msg
            )

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
                "TreatmentScore":   lambda a: (
                                        a.pain - a.addiction_score * ADDICTION_PENALTY_WEIGHT
                                    )                                  if isinstance(a, PatientAgent) else None,
                "ProfileType":      lambda a: a.profile_type           if isinstance(a, PatientAgent) else None,
                "AddictionMode":    lambda a: a.addiction_dynamic.mode if isinstance(a, PatientAgent) else None,
                "PainAUC":          lambda a: a.pain_auc               if isinstance(a, PatientAgent) else None,
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


def plot_results(model, df, n_steps):
    patient_ids = [a.unique_id for a in model.agents_by_type[PatientAgent]]

    # Build a lookup: agent unique_id → PatientAgent instance
    agent_lookup = {a.unique_id: a for a in model.agents_by_type[PatientAgent]}

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
        f"Doctor-Patient Simulation v6 — Path-Dependent Pain (α={ALPHA}) + "
        f"Pluggable Addiction Decay\n"
        f"treatment_score = pain − addiction × {ADDICTION_PENALTY_WEIGHT}",
        fontsize=13, fontweight="bold",
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
        auc_final   = pdata["PainAUC"].dropna().iloc[-1] if not pdata["PainAUC"].dropna().empty else 0.0
        profile     = pdata["ProfileType"].dropna().iloc[0]   if not pdata["ProfileType"].dropna().empty   else "?"
        addict_mode = pdata["AddictionMode"].dropna().iloc[0] if not pdata["AddictionMode"].dropna().empty else "?"

        profile_color = PROFILE_COLORS.get(profile, "#333333")
        addict_color  = ADDICTION_MODE_COLORS.get(addict_mode, "#555555")

        # ── Retrieve stored profile targets from the agent instance ──────
        # profile_targets[i] is the raw profile value at step i.
        # The datacollector calls collect() *before* PatientAgent.step(), so
        # steps[0] corresponds to the initial state (before any step), while
        # profile_targets[0] is recorded during step 0.  We therefore align
        # the target series to steps[1:] (one step ahead), padding the first
        # entry with base_pain so the arrays remain the same length.
        agent           = agent_lookup[pid]
        raw_targets     = agent.profile_targets          # length == n_steps
        base_pain_val   = agent.base_pain

        # Align: prepend base_pain as step-0 placeholder so len == len(steps)
        aligned_targets = [base_pain_val] + raw_targets[: len(steps) - 1]

        # ── Pain (with AUC annotation and profile-target overlay) ────────
        # 1. Faint profile-target dashed line (drawn first, below everything)
        ax_pain.plot(
            steps, aligned_targets,
            color=profile_color, linewidth=1.6, linestyle="--",
            alpha=0.55, zorder=1, label="Profile target",
        )

        # 2. Thin continuous actual-pain line for visual continuity
        ax_pain.plot(steps, pain_vals, color=profile_color, linewidth=1.2, alpha=0.4, zorder=2)

        # 3. Coloured segment lines keyed to treatment tier
        for i in range(len(steps) - 1):
            ax_pain.plot(
                steps[i:i+2], pain_vals[i:i+2],
                color=TIER_COLORS.get(tier_vals[i], "#888"),
                linewidth=2.2, zorder=3,
            )

        # 4. Scatter dots keyed to tier
        ax_pain.scatter(
            steps, pain_vals,
            c=[TIER_COLORS.get(t, "#888") for t in tier_vals],
            s=40, zorder=5,
        )

        # 5. Fill under actual pain
        ax_pain.fill_between(steps, pain_vals, alpha=0.07, color=profile_color, zorder=0)

        # 6. Background tier bands and threshold lines
        ax_pain.axhspan(0, 3,  color="#4CAF50", alpha=0.06)
        ax_pain.axhspan(3, 6,  color="#FF9800", alpha=0.06)
        ax_pain.axhspan(6, 10, color="#F44336", alpha=0.06)
        ax_pain.axhline(MED_SCORE_THRESHOLD,  color="#FF9800", lw=0.8, ls="--", alpha=0.5)
        ax_pain.axhline(HIGH_SCORE_THRESHOLD, color="#F44336",  lw=0.8, ls="--", alpha=0.5)

        ax_pain.set_ylim(0, 10)
        ax_pain.set_ylabel("Pain (0–10)")
        ax_pain.set_title(f"Patient {pid} — {profile}\nPain & Tier  |  AUC = {auc_final:.1f}")

        # Legend: tier colour patches + profile-target line
        tier_patches = [mpatches.Patch(color=c, label=t) for t, c in TIER_COLORS.items() if t]
        target_line  = plt.Line2D(
            [0], [0],
            color=profile_color, linewidth=1.6, linestyle="--", alpha=0.7,
            label="Profile target",
        )
        ax_pain.legend(handles=tier_patches + [target_line], fontsize=7, loc="upper right")

        # ── Addiction score ──
        ax_addict.plot(steps, addict_vals, color=addict_color, linewidth=2.0, label=f"mode: {addict_mode}")
        ax_addict.fill_between(steps, addict_vals, alpha=0.15, color=addict_color)
        ax_addict.set_ylabel("Addiction score")
        ax_addict.set_title(f"Patient {pid} — Addiction Score ({addict_mode})")
        ax_addict.legend(fontsize=7, loc="upper right")

        # ── Treatment score ──
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
    out_path = "simulation_plots_v6.png"
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
        addiction_decay_mode=None,
    )

    print("=== Doctor-Patient Simulation v6 (Pluggable Addiction Decay + Pain AUC) ===\n")
    for _ in range(N_STEPS):
        model.step()

    df = model.datacollector.get_agent_vars_dataframe()

    print("\n=== Patient Data (last 15 rows) ===")
    patient_df = df[df["Pain"].notna()]
    print(
        patient_df[[
            "Pain", "Tier", "ProfileType", "AddictionMode",
            "AddictionScore", "TreatmentScore", "PainAUC",
            "MediumTreatments", "HighTreatments", "Downgrades",
        ]]
        .tail(15)
        .to_string()
    )

    doctor = next(iter(model.agents_by_type[DoctorAgent]))
    print(f"\nTotal treatments given     : {doctor.treatments_given}")
    print(f"Addiction-driven downgrades: {doctor.downgrades_issued}")

    print("\n=== Per-Patient Summary ===")
    max_theoretical_auc = 10.0 * N_STEPS
    for a in model.agents_by_type[PatientAgent]:
        normalised = a.pain_auc / max_theoretical_auc * 100
        print(
            f"Patient {a.unique_id:>2} | profile={a.profile_type:<12} "
            f"decay={a.addiction_dynamic.mode:<8} | "
            f"medium={a.medium_treatment_count:>2}  high={a.high_treatment_count:>2}  "
            f"downgrades={a.downgrades_received:>2}  "
            f"final_addiction={a.addiction_score:>5.2f}  "
            f"pain_AUC={a.pain_auc:>7.2f}  ({normalised:.1f}% of max)"
        )

    plot_results(model, df, N_STEPS)