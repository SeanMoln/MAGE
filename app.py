"""
MARS-PMB 命令列工具

用法：
  python app.py preprocess --sample 100 [--seed 42] [--min-words 150] [--icd-top-n 5] [--output NAME]
  python app.py run-mars   [--input FILE] [--output FILE] [--cosine 92] [--jaccard 0.60] [--clear-memory]
  python app.py run-mage   [--input FILE] [--output FILE] [--cosine 92] [--jaccard 0.60]
  python app.py evaluate   --json FILE [--preprocessed FILE] [--mode hybrid]
                           [--comp-limit 0.65] [--lambda 0.6]
                           [--theta-facet 0.6] [--theta-element 0.6]
                           [--tau-facet 0.60] [--tau-element 0.62]
                           [--report FILE] [--csv FILE]
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


# ── 共用：找最新檔 ─────────────────────────────────────────────────────────────
def _latest(*patterns: tuple[str, str]) -> Path | None:
    """從多組 (directory, glob) 中回傳最新的匹配路徑。"""
    candidates = []
    for directory, pattern in patterns:
        p = Path(directory)
        if p.exists():
            candidates.extend(p.glob(pattern))
    return max(candidates, key=lambda f: f.name) if candidates else None


# ── Pipeline A: MARS 原始 ──────────────────────────────────────────────────────
def _run_mars(input_file="", output_file="", cosine=92.0, jaccard=0.60, clear_memory=False, limit=0):
    import json, time
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from agents.ManagerAgent import ManagerAgent
    from agents.PromptMemoryAgent import PromptMemoryAgent

    COSINE_THRESHOLD  = cosine
    JACCARD_THRESHOLD = jaccard

    print("="*70)
    print("🚀 MARS-PMB Pipeline A — MARS 原始")
    print(f"   cosine≥{COSINE_THRESHOLD}%  jaccard≥{JACCARD_THRESHOLD}")
    print("="*70)
    print()

    if clear_memory:
        subprocess.run([sys.executable, "-m", "src.utils.promtmemoryReset"],
                       cwd=str(Path(__file__).parent))
        print("記憶庫已清空。\n")

    if input_file and Path(input_file).exists():
        json_path = Path(input_file)
    else:
        candidates = sorted(Path("data/preprocessing").glob("discharge_preprocessed_*.json"))
        json_path = candidates[-1] if candidates else None

    if not json_path or not json_path.exists():
        print("❌ 找不到輸入檔案，請以 --input 指定路徑。")
        return 1

    with json_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    if limit > 0:
        cases = cases[:limit]

    print(f"【輸入資料】")
    print(f"  檔案: {json_path}")
    print(f"  案例數: {len(cases)}")
    print()

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        today = datetime.now().strftime("%Y%m%d")
        output_dir = Path("data/mars")
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(output_dir.glob(f"results_{today}_*.json"))
        seq = int(existing[-1].stem.rsplit("_", 1)[-1]) + 1 if existing else 1
        output_path = output_dir / f"results_{today}_{seq:03d}.json"

    print(f"【輸出路徑】  {output_path}\n")

    high_sim_count = 0
    similarity_list = []
    jaccard_list = []
    results = []
    task = "Summarize the following clinical note into a concise SOAP-style summary."

    for idx, input_text in enumerate(cases):
        print(f"\n{'='*70}")
        print(f"📋 案例 {idx + 1}/{len(cases)}  ({len(input_text.split())} 詞)")
        print(f"{'='*70}")

        manager = ManagerAgent()
        memory  = PromptMemoryAgent()

        similar_cases = memory.query_similar_cases(input_text=input_text, doc_type="SOAP", top_k=1)

        if similar_cases:
            best_sim  = similar_cases[0]["similarity"]
            ent_input = memory._extract_entities(input_text)
            ent_mem   = set(map(str.lower, similar_cases[0].get("entities", [])))
            jacc      = memory._jaccard(ent_input, ent_mem)
            similarity_list.append(best_sim)
            jaccard_list.append(jacc)
            triggered = best_sim >= COSINE_THRESHOLD and jacc >= JACCARD_THRESHOLD
            if triggered:
                high_sim_count += 1
                print(f"  🎯 高相似度  sim={best_sim:.2f}%  jaccard={jacc:.4f}  → 2 步優化")
            else:
                print(f"  🆕 新案例    sim={best_sim:.2f}%  jaccard={jacc:.4f}  → 4 步優化")
        else:
            best_sim, jacc = 0.0, 0.0
            triggered = False
            print(f"  🆕 全新案例（記憶庫為空）→ 4 步優化")

        optimized_prompt, final_result, error_msg = None, None, None
        for attempt in range(2):
            try:
                optimized_prompt, final_result = manager.run(task, input_text)
                break
            except Exception as e:
                error_msg = str(e)
                if attempt == 0:
                    print(f"  ⚠️ 錯誤: {e}，重試中...")
                    time.sleep(5)
                    manager = ManagerAgent()
                else:
                    print(f"  ❌ 失敗: {e}")

        results.append({
            "case_id":            idx,
            "input_text":         input_text,
            "input_length":       len(input_text.split()),
            "optimized_prompt":   optimized_prompt or "",
            "final_result":       final_result or f"[ERROR] {error_msg}",
            "error":              error_msg,
            "similarity":         best_sim,
            "jaccard":            jacc,
            "high_sim_triggered": triggered,
        })

        if not error_msg:
            usage = manager.get_token_usage()
            print(f"  token: prompt={usage['prompt_tokens']} "
                  f"completion={usage['completion_tokens']} total={usage['total_tokens']}")

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 統計
    print(f"\n{'='*70}")
    print("📊 執行統計")
    print(f"  通過雙門檻: {high_sim_count}/{len(cases)} ({high_sim_count/len(cases)*100:.1f}%)")
    if similarity_list:
        print(f"  相似度 avg={sum(similarity_list)/len(similarity_list):.2f}%"
              f"  ≥{COSINE_THRESHOLD:.0f}%: {sum(1 for s in similarity_list if s >= COSINE_THRESHOLD)} 筆")
    if jaccard_list:
        print(f"  Jaccard  avg={sum(jaccard_list)/len(jaccard_list):.4f}"
              f"  ≥{JACCARD_THRESHOLD}: {sum(1 for j in jaccard_list if j >= JACCARD_THRESHOLD)} 筆")
    print(f"  輸出: {output_path}  ({len(results)} 筆)")
    print(f"{'='*70}")
    return 0


# ── Pipeline B: MAGE ───────────────────────────────────────────────────────────
def _run_mage(input_file="", output_file="", cosine=92.0, jaccard=0.60, limit=0):
    import json, time
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from agents.ManagerAgent import ManagerAgent
    from agents.PromptMemoryAgent import PromptMemoryAgent

    COSINE_THRESHOLD  = cosine
    JACCARD_THRESHOLD = jaccard

    print("=" * 70)
    print("🔁 MARS-PMB Pipeline B — MAGE 缺漏補充")
    print("   Stage 1: TargetAgent → Pre_Draft 初稿")
    print("   Stage 2: MAGE Gap-detection → 補充缺漏")
    print(f"   cosine≥{COSINE_THRESHOLD}%  jaccard≥{JACCARD_THRESHOLD}")
    print("=" * 70)
    print()

    if input_file and Path(input_file).exists():
        json_path = Path(input_file)
    else:
        candidates = sorted(Path("data/preprocessing").glob("discharge_preprocessed_*.json"))
        json_path = candidates[-1] if candidates else None

    if not json_path or not json_path.exists():
        print("❌ 找不到輸入檔案，請以 --input 指定路徑。")
        return 1

    with json_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    if limit > 0:
        cases = cases[:limit]

    print(f"【輸入資料】")
    print(f"  檔案: {json_path}")
    print(f"  案例數: {len(cases)}")
    print()

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        today = datetime.now().strftime("%Y%m%d")
        output_dir = Path("data/mars")
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(output_dir.glob(f"mage_{today}_*.json"))
        seq = int(existing[-1].stem.rsplit("_", 1)[-1]) + 1 if existing else 1
        output_path = output_dir / f"mage_{today}_{seq:03d}.json"

    print(f"【輸出路徑】  {output_path}\n")

    task = "Summarize the following clinical note into a concise SOAP-style summary."
    results = []
    high_sim_count = 0
    similarity_list = []
    jaccard_list = []

    for idx, input_text in enumerate(cases):
        print(f"\n{'=' * 70}")
        print(f"📋 案例 {idx + 1}/{len(cases)}  ({len(input_text.split())} 詞)")
        print(f"{'=' * 70}")

        manager = ManagerAgent()
        memory  = PromptMemoryAgent()

        similar_cases = memory.query_similar_cases(input_text=input_text, doc_type="SOAP", top_k=1)

        best_sim, jacc, triggered = 0.0, 0.0, False
        if similar_cases:
            best_sim  = similar_cases[0]["similarity"]
            ent_input = memory._extract_entities(input_text)
            ent_mem   = set(map(str.lower, similar_cases[0].get("entities", [])))
            jacc      = memory._jaccard(ent_input, ent_mem)
            similarity_list.append(best_sim)
            jaccard_list.append(jacc)
            triggered = best_sim >= COSINE_THRESHOLD and jacc >= JACCARD_THRESHOLD
            if triggered:
                high_sim_count += 1
                print(f"  🎯 高相似度  sim={best_sim:.2f}%  jaccard={jacc:.4f}")
            else:
                print(f"  🆕 新案例    sim={best_sim:.2f}%  jaccard={jacc:.4f}")
        else:
            print(f"  🆕 全新案例（記憶庫為空）")

        optimized_prompt, final_result, pre_draft_result, error_msg = None, None, None, None
        for attempt in range(2):
            try:
                optimized_prompt, final_result, pre_draft_result = manager.run_mars(task, input_text)
                break
            except Exception as e:
                error_msg = str(e)
                if attempt == 0:
                    print(f"  ⚠️ 錯誤: {e}，重試中...")
                    time.sleep(5)
                    manager = ManagerAgent()
                else:
                    print(f"  ❌ 失敗: {e}")

        results.append({
            "case_id":            idx,
            "input_text":         input_text,
            "input_length":       len(input_text.split()),
            "pre_draft_result":   pre_draft_result or f"[ERROR] {error_msg}",
            "pre_draft_length":   len((pre_draft_result or "").split()),
            "optimized_prompt":   optimized_prompt or "",
            "final_result":       final_result or f"[ERROR] {error_msg}",
            "final_length":       len((final_result or "").split()),
            "error":              error_msg,
            "similarity":         best_sim,
            "jaccard":            jacc,
            "high_sim_triggered": triggered,
        })

        if not error_msg:
            usage = manager.get_token_usage()
            print(f"  token: prompt={usage['prompt_tokens']} "
                  f"completion={usage['completion_tokens']} total={usage['total_tokens']}")

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 統計
    print(f"\n{'=' * 70}")
    print("📊 執行統計")
    print(f"  通過雙門檻: {high_sim_count}/{len(cases)} ({high_sim_count/len(cases)*100:.1f}%)")
    if similarity_list:
        print(f"  相似度 avg={sum(similarity_list)/len(similarity_list):.2f}%"
              f"  ≥{COSINE_THRESHOLD:.0f}%: {sum(1 for s in similarity_list if s >= COSINE_THRESHOLD)} 筆")
    if jaccard_list:
        print(f"  Jaccard  avg={sum(jaccard_list)/len(jaccard_list):.4f}"
              f"  ≥{JACCARD_THRESHOLD}: {sum(1 for j in jaccard_list if j >= JACCARD_THRESHOLD)} 筆")

    predraft_lens = [r["pre_draft_length"] for r in results if r["pre_draft_length"] > 0]
    final_lens    = [r["final_length"]     for r in results if r["final_length"] > 0]
    orig_lens     = [r["input_length"]     for r in results]
    if predraft_lens and final_lens and orig_lens:
        avg_o = sum(orig_lens)/len(orig_lens)
        avg_p = sum(predraft_lens)/len(predraft_lens)
        avg_f = sum(final_lens)/len(final_lens)
        print(f"  原文 avg={avg_o:.0f}詞  Pre_Draft avg={avg_p:.0f}詞({avg_p/avg_o:.3f})"
              f"  MAGE avg={avg_f:.0f}詞({avg_f/avg_o:.3f})")

    print(f"  輸出: {output_path}  ({len(results)} 筆)")
    print(f"{'=' * 70}")
    return 0


# ── 預處理 ─────────────────────────────────────────────────────────────────────
def _run_preprocess(sample=100, seed=42, min_words=150, icd_top_n=5, output=""):
    before = set(Path("data/preprocessing").glob("discharge_preprocessed_*.json")) \
             if Path("data/preprocessing").exists() else set()

    rc = subprocess.run(
        [sys.executable, "preprocessing/src/extract_discharge.py",
         "--sample",    str(sample),
         "--seed",      str(seed),
         "--min_words", str(min_words),
         "--icd_top_n", str(icd_top_n)],
        cwd=str(Path(__file__).parent),
    ).returncode

    if rc == 0 and output.strip():
        after = set(Path("data/preprocessing").glob("discharge_preprocessed_*.json"))
        new_files = after - before
        if new_files:
            name = output.strip()
            if not name.endswith(".json"):
                name += ".json"
            next(iter(new_files)).rename(Path("data/preprocessing") / name)
            print(f"輸出已重新命名為: data/preprocessing/{name}")

    return rc


# ── 評估 ───────────────────────────────────────────────────────────────────────
def _run_evaluate(json_file, preprocessed="", mode="hybrid", comp_limit=0.65,
                  lambda_val=0.6, theta_facet=0.6, theta_element=0.6,
                  tau_facet=0.60, tau_element=0.62, report="", csv=""):
    eval_dir = Path("data/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")

    if report.strip():
        _r = report.strip()
        out_report = Path(_r) if "/" in _r or "\\" in _r else eval_dir / (_r if _r.endswith(".json") else _r + ".json")
    else:
        existing = sorted(eval_dir.glob(f"report_{today}_*.json"))
        seq = int(existing[-1].stem.rsplit("_", 1)[-1]) + 1 if existing else 1
        out_report = eval_dir / f"report_{today}_{seq:03d}.json"

    if csv.strip():
        _c = csv.strip()
        out_csv = Path(_c) if "/" in _c or "\\" in _c else eval_dir / (_c if _c.endswith(".csv") else _c + ".csv")
    else:
        existing_csv = sorted(eval_dir.glob(f"metrics_{today}_*.csv"))
        seq_csv = int(existing_csv[-1].stem.rsplit("_", 1)[-1]) + 1 if existing_csv else 1
        out_csv = eval_dir / f"metrics_{today}_{seq_csv:03d}.csv"

    cmd = [
        sys.executable, "evaluation/evalute_summary.py",
        "--json",           json_file,
        "--csv",            str(out_csv),
        "--mode",           mode,
        "--lambda_facet",   str(lambda_val),
        "--lambda_element", str(lambda_val),
        "--theta_facet",    str(theta_facet),
        "--theta_element",  str(theta_element),
        "--tau_facet",      str(tau_facet),
        "--tau_element",    str(tau_element),
        "--comp_limit",     str(comp_limit),
        "--avg",
        "--out", str(out_report),
    ]
    if preprocessed.strip():
        cmd += ["--preprocessed", preprocessed.strip()]

    return subprocess.run(cmd, cwd=str(Path(__file__).parent)).returncode


# ── CLI 進入點 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="MARS-PMB 命令列工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    # ── preprocess ────────────────────────────────────────────────────────────
    pp = sub.add_parser("preprocess", help="MIMIC-III 資料預處理")
    pp.add_argument("--sample",    type=int, default=100,  help="抽取筆數（預設 100）")
    pp.add_argument("--seed",      type=int, default=42,   help="隨機種子（預設 42）")
    pp.add_argument("--min-words", type=int, default=150,  dest="min_words", help="最小字數門檻（預設 150）")
    pp.add_argument("--icd-top-n", type=int, default=5,    dest="icd_top_n", help="ICD 碼取前 N 碼（預設 5）")
    pp.add_argument("--output",    default="",             help="輸出檔名（預設自動生成）")

    # ── run-mars ──────────────────────────────────────────────────────────────
    rm = sub.add_parser("run-mars", help="Pipeline A：MARS 原始（Teacher-Critic-Student + PromptMemory）")
    rm.add_argument("--input",        default="",   dest="input_file",  help="輸入 JSON 路徑（預設讀最新預處理檔）")
    rm.add_argument("--output",       default="",   dest="output_file", help="輸出 JSON 路徑（預設自動生成）")
    rm.add_argument("--cosine",       type=float,   default=92.0,       help="cosine 門檻 %%（預設 92）")
    rm.add_argument("--jaccard",      type=float,   default=0.60,       help="Jaccard 門檻（預設 0.60）")
    rm.add_argument("--clear-memory", action="store_true", dest="clear_memory", help="執行前清空 prompt memory")
    rm.add_argument("--limit",   type=int, default=0, help="只跑前 N 筆（0 = 全部）")

    # ── run-mage ──────────────────────────────────────────────────────────────
    mg = sub.add_parser("run-mage", help="Pipeline B：MAGE（Pre_Draft → Gap-detection 缺漏補充）")
    mg.add_argument("--input",   default="",  dest="input_file",  help="輸入 JSON 路徑（預設讀最新預處理檔）")
    mg.add_argument("--output",  default="",  dest="output_file", help="輸出 JSON 路徑（預設自動生成）")
    mg.add_argument("--cosine",  type=float,  default=92.0,       help="cosine 門檻 %%（預設 92）")
    mg.add_argument("--jaccard", type=float,  default=0.60,       help="Jaccard 門檻（預設 0.60）")
    mg.add_argument("--limit",   type=int,    default=0,          help="只跑前 N 筆（0 = 全部）")

    # ── evaluate ──────────────────────────────────────────────────────────────
    ev = sub.add_parser("evaluate", help="評估摘要品質（SOAP retention + Key Element Recall）")
    ev.add_argument("--json",          required=True,  dest="json_file",    help="MARS/MAGE 結果 JSON")
    ev.add_argument("--preprocessed",  default="",                           help="完整原文 JSON（選填）")
    ev.add_argument("--mode",          default="hybrid",
                    choices=["hybrid", "rule", "semantic"],                  help="評估模式（預設 hybrid）")
    ev.add_argument("--comp-limit",    type=float, default=0.65, dest="comp_limit",
                                                                             help="壓縮率上限（預設 0.65）")
    ev.add_argument("--lambda",        type=float, default=0.6,  dest="lambda_val",
                                                                             help="λ rule 比重，hybrid 用（預設 0.6）")
    ev.add_argument("--theta-facet",   type=float, default=0.6,  dest="theta_facet")
    ev.add_argument("--theta-element", type=float, default=0.6,  dest="theta_element")
    ev.add_argument("--tau-facet",     type=float, default=0.60, dest="tau_facet")
    ev.add_argument("--tau-element",   type=float, default=0.62, dest="tau_element")
    ev.add_argument("--report",        default="",                            help="報告 JSON 路徑（預設自動生成）")
    ev.add_argument("--csv",           default="",                            help="指標 CSV 路徑（預設自動生成）")

    args = parser.parse_args()

    if args.command == "preprocess":
        sys.exit(_run_preprocess(
            sample=args.sample, seed=args.seed,
            min_words=args.min_words, icd_top_n=args.icd_top_n,
            output=args.output,
        ))

    elif args.command == "run-mars":
        sys.exit(_run_mars(
            input_file=args.input_file, output_file=args.output_file,
            cosine=args.cosine, jaccard=args.jaccard,
            clear_memory=args.clear_memory, limit=args.limit,
        ))

    elif args.command == "run-mage":
        sys.exit(_run_mage(
            input_file=args.input_file, output_file=args.output_file,
            cosine=args.cosine, jaccard=args.jaccard, limit=args.limit,
        ))

    elif args.command == "evaluate":
        sys.exit(_run_evaluate(
            json_file=args.json_file, preprocessed=args.preprocessed,
            mode=args.mode, comp_limit=args.comp_limit,
            lambda_val=args.lambda_val,
            theta_facet=args.theta_facet, theta_element=args.theta_element,
            tau_facet=args.tau_facet, tau_element=args.tau_element,
            report=args.report, csv=args.csv,
        ))


if __name__ == "__main__":
    main()
