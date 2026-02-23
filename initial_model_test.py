import pandas as pd
import numpy as np
import mesa
import seaborn as sns
import random
import math


# ─────────────────────────── Agents ───────────────────────────

class PatientAgent(mesa.Agent):
    """A patient whose pain oscillates and can be reduced by treatment."""

    def __init__(self, unique_id, model, base_pain: float = 5.0):
        super().__init__(unique_id, model)
        self.base_pain = base_pain          # baseline pain level (0–10)
        self.pain: float = base_pain        # current pain level
        self.treated_this_step: bool = False

    # ---------- pain function ----------
    def _natural_pain_evolution(self) -> float:
        """
        Pain oscillates sinusoidally around the baseline and adds a small
        random perturbation each step.
        """
        t = self.model.schedule.steps
        oscillation = 2.0 * math.sin(t / 5.0)          # periodic component
        noise = random.uniform(-0.5, 0.5)               # random noise
        new_pain = self.base_pain + oscillation + noise
        return max(0.0, min(10.0, new_pain))            # clamp to [0, 10]

    # ---------- step ----------
    def step(self):
        # Natural pain evolution
        self.pain = self._natural_pain_evolution()
        self.treated_this_step = False          # reset flag each step

    def receive_treatment(self, reduction: float = 3.0):
        """Doctor calls this to reduce the patient's pain."""
        self.pain = max(0.0, self.pain - reduction)
        self.treated_this_step = True
        print(f"  [Step {self.model.schedule.steps}] Patient {self.unique_id} "
              f"treated → pain reduced to {self.pain:.2f}")


class DoctorAgent(mesa.Agent):
    """
    A doctor who monitors patients and administers treatment when:
      - patient pain ≥ pain_threshold
      - doctor has not exceeded their daily treatment quota (max_treatments_per_step)
    """

    def __init__(
        self,
        unique_id,
        model,
        pain_threshold: float = 6.0,
        max_treatments_per_step: int = 1,
        treatment_reduction: float = 3.0,
    ):
        super().__init__(unique_id, model)
        self.pain_threshold = pain_threshold            # min pain to trigger treatment
        self.max_treatments_per_step = max_treatments_per_step  # doctor restriction
        self.treatment_reduction = treatment_reduction  # how much pain is reduced
        self.treatments_given: int = 0                  # total treatments given

    def step(self):
        treatments_this_step = 0

        patients = [
            a for a in self.model.schedule.agents
            if isinstance(a, PatientAgent)
        ]

        # Sort by pain descending so the most painful patient is treated first
        patients.sort(key=lambda p: p.pain, reverse=True)

        for patient in patients:
            # Check doctor restriction (quota)
            if treatments_this_step >= self.max_treatments_per_step:
                print(f"  [Step {self.model.schedule.steps}] Doctor {self.unique_id} "
                      f"reached treatment quota ({self.max_treatments_per_step}).")
                break

            if patient.pain >= self.pain_threshold:
                patient.receive_treatment(self.treatment_reduction)
                treatments_this_step += 1
                self.treatments_given += 1
            else:
                print(f"  [Step {self.model.schedule.steps}] Patient {patient.unique_id} "
                      f"pain={patient.pain:.2f} below threshold – no treatment.")


# ─────────────────────────── Model ────────────────────────────

class DoctorPatientModel(mesa.Model):
    """Simple model containing one doctor and one (or more) patients."""

    def __init__(
        self,
        n_patients: int = 1,
        pain_threshold: float = 6.0,
        max_treatments_per_step: int = 1,
    ):
        super().__init__()
        self.schedule = mesa.time.RandomActivation(self)

        # Create patients
        for i in range(n_patients):
            base_pain = random.uniform(3.0, 7.0)
            patient = PatientAgent(unique_id=i, model=self, base_pain=base_pain)
            self.schedule.add(patient)

        # Create one doctor
        doctor = DoctorAgent(
            unique_id=n_patients,
            model=self,
            pain_threshold=pain_threshold,
            max_treatments_per_step=max_treatments_per_step,
        )
        self.schedule.add(doctor)

        # Data collector
        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Pain": lambda a: a.pain if isinstance(a, PatientAgent) else None,
                "Treated": lambda a: a.treated_this_step if isinstance(a, PatientAgent) else None,
                "TotalTreatmentsGiven": lambda a: a.treatments_given if isinstance(a, DoctorAgent) else None,
            }
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


# ─────────────────────────── Run ──────────────────────────────

if __name__ == "__main__":
    N_STEPS = 20

    model = DoctorPatientModel(
        n_patients=2,
        pain_threshold=6.0,
        max_treatments_per_step=1,   # doctor can only treat 1 patient per step
    )

    print("=== Doctor-Patient Simulation ===\n")
    for _ in range(N_STEPS):
        model.step()

    # Summary
    df = model.datacollector.get_agent_vars_dataframe()
    print("\n=== Agent Data (last 5 steps) ===")
    print(df.tail(10).to_string())

    doctor = next(a for a in model.schedule.agents if isinstance(a, DoctorAgent))
    print(f"\nTotal treatments given by doctor: {doctor.treatments_given}")