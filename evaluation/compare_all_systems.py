"""
四組系統統計比較：
  1. Simple Baseline     data/baseline/baseline_simple_results.json
  2. Baseline            data/baseline/baseline_20260326_002.json
  3. MARS (原版)          data/mars/results_20260326_002.json
  4. MARS (TeacherFixed) data/mars/mars_teacher_fixed_100.json

輸出：
  - 各組 Macro Mean ± SD
  - Wilcoxon signed-rank test（配對，相同 case_id）
  - data/evaluation/compare_all_systems.json
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy import stats

# 讓 evaluate_pair 可以被 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.evalute_summary import evaluate_pair

KE_KEYS = ["diagnoses", "labs", "imaging", "comorbidities", "plan_general"]

SYSTEMS = {
    "Simple":    "data/baseline/baseline_simple_results.json",
    "Baseline":  "data/baseline/baseline_20260326_002.json",
    "MARS_orig": "data/mars/results_20260326_002.json",
    "MARS_tf":   "data/mars/mars_teacher_fixed_100.json",
}


def load_and_eval(path_str: str, comp_limit: float = 0.65) -> dict:
    """讀取結果 JSON，逐筆評估，回傳 {case_id: EvalReport}"""
    path = Path(path_str)
    if not path.exists():
        print(f"  [SKIP] 找不到: {path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    for rec in data:
        cid     = rec.get("case_id", rec.get("id"))
        summary = rec.get("final_result", "")
        inp     = rec.get("input_text", "")
        inp_len = rec.get("input_length", len(inp.split()))

        if not summary or summary.startswith("[ERROR]"):
            continue
        try:
            report = evaluate_pair(
                original_soap_text=inp,
                bullet_summary_text=summary,
                mode="rule",
                comp_limit=comp_limit,
                w_orig_override=inp_len,
            )
            results[cid] = report
        except Exception as e:
            print(f"  評估錯誤 case {cid}: {e}")

    return results


def ke_avg_val(report) -> float:
    vals = [report.key_element_recall.get(k, 1.0) for k in KE_KEYS]
    return float(np.mean(vals))


def soap_avg_val(report) -> float:
    sr = report.soap_retention
    return float(np.mean([sr.get(k, 0.0) for k in ["S", "O", "A", "P"]]))


def get_series(eval_dict: dict, metric: str) -> dict:
    """回傳 {case_id: float}"""
    out = {}
    for cid, r in eval_dict.items():
        if metric == "ke_avg":
            out[cid] = ke_avg_val(r)
        elif metric == "soap_avg":
            out[cid] = soap_avg_val(r)
        elif metric in KE_KEYS:
            out[cid] = r.key_element_recall.get(metric, 1.0)
        elif metric == "compression":
            out[cid] = r.compression_ratio
    return out


def wilcoxon(a: list, b: list):
    diffs = [x - y for x, y in zip(a, b) if x != y]
    if len(diffs) < 10:
        return None, None
    try:
        s, p = stats.wilcoxon(diffs)
        return float(s), float(p)
    except Exception:
        return None, None


def sig_str(p):
    if p is None:
        return "N/A"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def print_summary_table(all_evals: dict):
    metrics = ["ke_avg", "soap_avg"] + KE_KEYS + ["compression"]
    labels  = ["KE avg", "SOAP avg", "diagnoses", "labs",
                "imaging", "comorbidities", "plan_general", "compression"]
    names   = list(all_evals.keys())

    print("\n" + "=" * 90)
    print("指標比較表（Mean ± SD）")
    print("=" * 90)
    print(f"{'指標':<18}" + "".join(f"{n:>18}" for n in names))
    print("-" * 90)
    for m, lbl in zip(metrics, labels):
        row = f"{lbl:<18}"
        for name in names:
            vals = list(get_series(all_evals[name], m).values())
            if vals:
                row += f"  {np.mean(vals):.4f}±{np.std(vals):.4f}"
            else:
                row += f"{'N/A':>18}"
        print(row)
    print("=" * 90)


def print_pairwise(all_evals: dict, ref: str):
    if ref not in all_evals:
        return
    ref_ev  = all_evals[ref]
    metrics = ["ke_avg"] + KE_KEYS
    labels  = ["KE avg"] + KE_KEYS

    print(f"\n{'='*90}")
    print(f"Wilcoxon signed-rank test  vs  {ref}")
    print(f"{'='*90}")
    print(f"{'指標':<18}{'系統':<16}{'n 配對':>8}{'Δ mean':>10}{'p-value':>12}{'sig':>6}")
    print("-" * 90)

    for other, other_ev in all_evals.items():
        if other == ref:
            continue
        common = sorted(set(ref_ev) & set(other_ev))
        for m, lbl in zip(metrics, labels):
            rs = get_series(ref_ev,   m)
            os = get_series(other_ev, m)
            a  = [rs[c] for c in common if c in rs and c in os]
            b  = [os[c] for c in common if c in rs and c in os]
            n  = len(a)
            delta = np.mean(b) - np.mean(a) if a else float("nan")
            s, p  = wilcoxon(a, b)
            p_str = f"{p:.4f}" if p is not None else "N/A"
            print(f"{lbl:<18}{other:<16}{n:>8}{delta:>+10.4f}{p_str:>12}{sig_str(p):>6}")
        print()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp_limit", type=float, default=0.65)
    args = ap.parse_args()

    print("\n" + "=" * 90)
    print("四組系統全面比較")
    print("=" * 90)

    all_evals = {}
    for name, path in SYSTEMS.items():
        print(f"\n評估 {name} ...")
        ev = load_and_eval(path, comp_limit=args.comp_limit)
        print(f"  有效案例: {len(ev)}")
        all_evals[name] = ev

    print_summary_table(all_evals)
    print_pairwise(all_evals, ref="Baseline")
    print_pairwise(all_evals, ref="MARS_orig")

    # JSON 摘要
    summary = {}
    for name, ev in all_evals.items():
        if not ev:
            continue
        ke_vals = list(get_series(ev, "ke_avg").values())
        summary[name] = {
            "n":           len(ev),
            "ke_avg_mean": round(float(np.mean(ke_vals)), 4),
            "ke_avg_std":  round(float(np.std(ke_vals)),  4),
            "soap_avg":    round(float(np.mean(list(get_series(ev, "soap_avg").values()))), 4),
            **{k: round(float(np.mean(list(get_series(ev, k).values()))), 4) for k in KE_KEYS},
        }

    out = Path("data/evaluation/compare_all_systems.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nJSON 摘要: {out}")


if __name__ == "__main__":
    main()
