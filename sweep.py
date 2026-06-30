"""Sweep beberapa konfigurasi sekaligus (satu proses) di atas sampel yang sama."""
import time
import pandas as pd
import eval_classic as E

df = pd.read_csv("dataset/test_dataset/test.csv").sample(400, random_state=42).reset_index(drop=True)

B = dict(ED_PENALTY=4.0, PRIOR_W=0.3, MIN_LEN=3, W_EDIT=2.0, MARGIN=0.0, candgen="improved")
def cfg(**kw):
    d = dict(B); d.update(kw); return d

CONFIGS = {
    "baseline":           dict(ED_PENALTY=4.0, PRIOR_W=0.3, MIN_LEN=3, W_EDIT=2.0, MARGIN=0.0, candgen="baseline"),
    "proper":             cfg(SKIP_PROPER=True),
    "proper+minlen4":     cfg(SKIP_PROPER=True, MIN_LEN=4),
    "proper+ed6":         cfg(SKIP_PROPER=True, ED_PENALTY=6.0),
    "proper+floorSoft":   cfg(SKIP_PROPER=True, UNI_FLOOR=-12.0),
    "proper+floorMed":    cfg(SKIP_PROPER=True, UNI_FLOOR=-11.0),
}

E.VARIANTS.update(CONFIGS)
rows = []
for name in CONFIGS:
    t0 = time.time()
    m = E.evaluate(df, name)
    m["secs"] = round(time.time() - t0, 1)
    rows.append(m)
    print(f"done {name:18s} {m['secs']}s  P={m['Precision']:.3f} R={m['Recall']:.3f} "
          f"F1={m['F1']:.3f} candR={m['cand_recall']:.3f} Top5={m['Top5']:.3f}")

cols = ["variant", "Precision", "Recall", "F1", "Accuracy",
        "WER_reduction", "CER_reduction", "cand_recall", "Top1", "Top3", "Top5", "MRR",
        "FP", "FN"]
out = pd.DataFrame(rows)[cols].set_index("variant").round(4)
print("\n==================== RINGKASAN ====================")
print(out.to_string())
