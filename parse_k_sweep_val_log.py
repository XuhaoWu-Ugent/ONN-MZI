"""Parse the SLURM log of run_clean_k_sweep_val.sh into a per-K table.

Usage:
    python parse_k_sweep_val_log.py logs/clean_k_val-<jobid>.out [out.csv]

For each K it extracts the "[Final] ..." summary line written by
train_distill.py when --val-split > 0:

    [Final] K=15 val_split=5000 best_val_acc=87.12% at epoch 28 |
            test_acc_at_best_val=86.90% | last_epoch_test_acc=86.85%

and, for comparison with the legacy protocol, the maximum test accuracy seen
over all epochs of that K (the quantity plotted in the original Fig. S2c).
Columns: K, best_val_acc, best_val_epoch, test_acc_at_best_val,
last_epoch_test_acc, max_test_acc_legacy.
"""
import csv
import re
import sys

FINAL = re.compile(
    r"\[Final\] K=(\d+) val_split=(\d+) best_val_acc=([\d.]+)% at epoch (\d+) \| "
    r"test_acc_at_best_val=([\d.]+)% \| last_epoch_test_acc=([\d.]+)%")
START = re.compile(r">>> Starting Clean-FC K=(\d+)")
TEST = re.compile(r"^Test set \(Ensemble \d+x\): .*\(([\d.]+)%\)")


def parse(path):
    rows, cur_k, max_test = [], None, {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = START.search(line)
            if m:
                cur_k = int(m.group(1)); max_test.setdefault(cur_k, 0.0); continue
            m = TEST.match(line.strip())
            if m and cur_k is not None:
                max_test[cur_k] = max(max_test[cur_k], float(m.group(1))); continue
            m = FINAL.search(line)
            if m:
                k = int(m.group(1))
                rows.append({
                    "K": k,
                    "val_split": int(m.group(2)),
                    "best_val_acc": float(m.group(3)),
                    "best_val_epoch": int(m.group(4)),
                    "test_acc_at_best_val": float(m.group(5)),
                    "last_epoch_test_acc": float(m.group(6)),
                    "max_test_acc_legacy": max_test.get(k, float("nan")),
                })
    return sorted(rows, key=lambda r: r["K"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = parse(sys.argv[1])
    cols = ["K", "val_split", "best_val_acc", "best_val_epoch",
            "test_acc_at_best_val", "last_epoch_test_acc", "max_test_acc_legacy"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"wrote {sys.argv[2]}")
