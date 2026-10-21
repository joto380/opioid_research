# sobol_analysis.py
import copy
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

import initial_model_v9 as M

# ───────────────────────── PARAMETER SPACE ─────────────────────────
# Plain module-level scalars: name -> (low, high)
SCALARS = {
    "ALPHA": (0.05, 0.5),
    "ADDICTION_PENALTY_WEIGHT": (0.1, 0.6),
    "HIGH_SCORE_THRESHOLD": (4.5, 7.5),
    "MED_SCORE_THRESHOLD": (2.0, 4.0),
    "AUC_MEM_WINDOW": (5, 20),                      # rounded to int
    "ADDICTION_AUC_LOW_THRESHOLD": (1.0, 3.0),
    "ADDICTION_AUC_HIGH_THRESHOLD": (3.5, 7.0),
    "PAIN_CRITICAL_THRESHOLD": (7.0, 9.0),
    "PAIN_EXPLOSION_COEF": (1.0, 10.0),
    "PAIN_EXPLOSION_POWER": (2.0, 4.0),
    "ADDICTION_RISK_NOISE_STD": (0.0, 0.4),
    "TOLERANCE_GAIN_MEDIUM": (0.2, 0.6),
    "TOLERANCE_GAIN_HIGH": (0.4, 1.2),
    "TOLERANCE_DECAY_RATE": (0.02, 0.15),
    "TOLERANCE_50": (3.0, 8.0),
    "CRAVING_GAIN_PER_DOSE": (0.3, 1.0),
    "CRAVING_TOLERANCE_SCALE": (0.0, 0.3),
    "CRAVING_DECAY_RATE": (0.02, 0.2),
    "CRAVING_PAIN_SUPPRESSION": (0.0, 0.15),
    "FUNC_SIGMOID_MID": (0.3, 0.7),
    "FUNC_SIGMOID_K": (2.0, 8.0),
    # raw functionality weights: renormalized to sum to 1 after sampling
    "FUNC_WEIGHT_PAIN": (0.05, 0.5),
    "FUNC_WEIGHT_ADDICTION": (0.05, 0.5),
    "FUNC_WEIGHT_TOLERANCE": (0.05, 0.5),
    "FUNC_WEIGHT_CRAVING": (0.05, 0.5),
    "FUNC_WEIGHT_TREATMENT_TIER": (0.05, 0.5),
}
FUNC_WEIGHTS = [k for k in SCALARS if k.startswith("FUNC_WEIGHT_")]
INT_PARAMS = {"AUC_MEM_WINDOW"}

# Entries inside the nested dicts: name -> (dict name, key, subkey, (low, high))
DICT_PARAMS = {
    "TIER_LOW_reduction":      ("TIER_EFFECTS", "LOW", "reduction", (0.5, 2.5)),
    "TIER_MED_reduction":      ("TIER_EFFECTS", "MEDIUM", "reduction", (2.0, 4.0)),
    "TIER_HIGH_reduction":     ("TIER_EFFECTS", "HIGH", "reduction", (3.5, 6.5)),
    "TIER_MED_addiction_w":    ("TIER_EFFECTS", "MEDIUM", "addiction_weight", (0.5, 1.5)),
    "TIER_HIGH_addiction_w":   ("TIER_EFFECTS", "HIGH", "addiction_weight", (1.5, 4.5)),
    "ZONE1_reduction_penalty": ("ADDICTION_AUC_EFFECTS", 1, "reduction_penalty", (0.0, 1.0)),
    "ZONE2_reduction_penalty": ("ADDICTION_AUC_EFFECTS", 2, "reduction_penalty", (0.5, 2.0)),
    "ZONE1_addiction_incr":    ("ADDICTION_AUC_EFFECTS", 1, "addiction_increase", (0.0, 0.3)),
    "ZONE2_addiction_incr":    ("ADDICTION_AUC_EFFECTS", 2, "addiction_increase", (0.1, 0.6)),
}

# Policy argument (not a module constant)
POLICY = {"pain_threshold": (2.0, 5.0)}

NAMES = list(SCALARS) + list(DICT_PARAMS) + list(POLICY)
BOUNDS = ([list(v) for v in SCALARS.values()]
          + [list(v[3]) for v in DICT_PARAMS.values()]
          + [list(v) for v in POLICY.values()])
PROBLEM = {"num_vars": len(NAMES), "names": NAMES, "bounds": BOUNDS}

DOCTORS = ("SCORE", "FUNCTIONALITY", "RAW_PAIN")
PAIRS = [("SCORE", "RAW_PAIN"), ("FUNCTIONALITY", "RAW_PAIN"), ("FUNCTIONALITY", "SCORE")]
N_TRAJ, N_STEPS = 30, 30

OUTPUT_NAMES = (
    [f"{m}|{d}" for m in M.SUMMARY_METRICS for d in DOCTORS]
    + [f"{m}|{a}-{b}" for m in M.SUMMARY_METRICS for a, b in PAIRS]
)

# ───────────────────────── PATCHING ─────────────────────────
def _snapshot():
    keys = list(SCALARS) + ["TIER_EFFECTS", "ADDICTION_AUC_EFFECTS"]
    return {k: copy.deepcopy(getattr(M, k)) for k in keys}

def _restore(snap):
    for k, v in snap.items():
        setattr(M, k, v)

def _apply(p):
    scal = {k: p[k] for k in SCALARS}
    for k in INT_PARAMS:
        scal[k] = int(round(scal[k]))
    wsum = sum(scal[k] for k in FUNC_WEIGHTS)
    for k in FUNC_WEIGHTS:
        scal[k] /= wsum
    for k, v in scal.items():
        setattr(M, k, v)
    for name, (dname, key, sub, _) in DICT_PARAMS.items():
        getattr(M, dname)[key][sub] = p[name]

# ───────────────────────── EVALUATION ─────────────────────────
def evaluate(x):
    p = dict(zip(NAMES, x))
    snap = _snapshot()
    try:
        _apply(p)
        frames = {}
        for dt in DOCTORS:
            _, rows = M.run_batch_by_profile(N_TRAJ, N_STEPS, p["pain_threshold"], 3, doctor_type=dt)
            frames[dt] = pd.DataFrame(rows).set_index("run_idx")[M.SUMMARY_METRICS]
    finally:
        _restore(snap)

    out = []
    for m in M.SUMMARY_METRICS:
        out += [np.nanmean(frames[d][m]) for d in DOCTORS]
    for m in M.SUMMARY_METRICS:
        out += [np.nanmean(frames[a][m] - frames[b][m]) for a, b in PAIRS]   # paired by run_idx
    return np.nan_to_num(np.array(out, dtype=float), nan=0.0)

# ───────────────────────── MAIN ─────────────────────────
if __name__ == "__main__":
    N = 128   # must be a power of 2; total evals = N * (D + 2)
    X = sobol_sample.sample(PROBLEM, N, calc_second_order=False, seed=0)
    print(f"D={len(NAMES)}  N={N}  parameter sets={len(X)}  "
          f"model batches={len(X) * len(DOCTORS)}")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        Y = np.array(list(ex.map(evaluate, X, chunksize=4)))
    np.save("sobol_X.npy", X)
    np.save("sobol_Y.npy", Y)

    rows = []
    for j, oname in enumerate(OUTPUT_NAMES):
        if np.ptp(Y[:, j]) == 0:
            continue                                  # constant output, nothing to attribute
        Si = sobol_analyze.analyze(PROBLEM, Y[:, j], calc_second_order=False,
                                   num_resamples=200, conf_level=0.95, seed=0,
                                   print_to_console=False)
        for i, n in enumerate(NAMES):
            rows.append({"output": oname, "param": n,
                         "S1": Si["S1"][i], "S1_conf": Si["S1_conf"][i],
                         "ST": Si["ST"][i], "ST_conf": Si["ST_conf"][i]})
    res = pd.DataFrame(rows)
    res.to_csv("sobol_indices.csv", index=False)

    for target in ["pain_auc|FUNCTIONALITY-RAW_PAIN", "addiction_auc|FUNCTIONALITY-RAW_PAIN",
                   "pain_auc|SCORE-RAW_PAIN"]:
        sub = res[res.output == target].sort_values("ST", ascending=False)
        print(f"\n=== {target} ===")
        print(sub.head(12).round(3).to_string(index=False))