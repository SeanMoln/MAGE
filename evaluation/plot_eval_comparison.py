# plot_eval_comparison.py
# -*- coding: utf-8 -*-
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

# ============== 設定檔名和路徑 ==============
RESULTS_DIR = "data/evaluation"
RULE_CSV    = os.path.join(RESULTS_DIR, "metrics_rule.csv")
SEM_CSV     = os.path.join(RESULTS_DIR, "metrics_semantic.csv")
HYB_CSV     = os.path.join(RESULTS_DIR, "metrics_hybrid.csv")

# ============== 讀檔與檢查 ==============
missing = [p for p in [RULE_CSV, SEM_CSV, HYB_CSV] if not os.path.exists(p)]
if missing:
    print("找不到以下檔案：", missing)
    print("請先執行評估腳本：")
    print("  ./evaluate.sh all")
    print("或手動執行：")
    print("  python evaluation/evalute_summary.py --json data/mars/mimic_discharge_100_results.json --csv data/evaluation/metrics_rule.csv --mode rule --comp_limit 0.65 --avg")
    print("  python evaluation/evalute_summary.py --json data/mars/mimic_discharge_100_results.json --csv data/evaluation/metrics_semantic.csv --mode semantic --theta_facet 0.6 --theta_element 0.6 --tau_facet 0.60 --tau_element 0.62 --comp_limit 0.65 --avg --out data/evaluation/semantic_report.json")
    print("  python evaluation/evalute_summary.py --json data/mars/mimic_discharge_100_results.json --csv data/evaluation/metrics_hybrid.csv --mode hybrid --lambda_facet 0.6 --lambda_element 0.6 --theta_facet 0.6 --theta_element 0.6 --tau_facet 0.60 --tau_element 0.62 --comp_limit 0.65 --avg --out data/evaluation/hybrid_report.json")
    sys.exit(1)

print(f"讀取評估結果...")
print(f"  Rule-based: {RULE_CSV}")
print(f"  Semantic: {SEM_CSV}")
print(f"  Hybrid: {HYB_CSV}")

df_rule = pd.read_csv(RULE_CSV)
df_sem  = pd.read_csv(SEM_CSV)
df_hyb  = pd.read_csv(HYB_CSV)

# ============== 欄位名稱 ==============
facet_cols = ["S_retained","O_retained","A_retained","P_retained"]
elem_cols  = ["recall_diagnoses","recall_labs","recall_imaging","recall_comorbidities","recall_plan_general","recall_plan_minimums_adhf"]

# ============== 計算平均 ==============
def safe_mean(df, cols):
    return df[cols].mean(numeric_only=True)

avg_rule_f = safe_mean(df_rule, facet_cols)
avg_sem_f  = safe_mean(df_sem, facet_cols)
avg_hyb_f  = safe_mean(df_hyb, facet_cols)

avg_rule_e = safe_mean(df_rule, elem_cols)
avg_sem_e  = safe_mean(df_sem, elem_cols)
avg_hyb_e  = safe_mean(df_hyb, elem_cols)

# ============== 1) SOAP 構面平均比較（長條圖） ==============
labels = facet_cols
x = range(len(labels))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar([i - width for i in x], [avg_rule_f[c] for c in labels], width, label="Rule")
plt.bar([i           for i in x], [avg_sem_f[c]  for c in labels], width, label="Semantic")
plt.bar([i + width   for i in x], [avg_hyb_f[c]  for c in labels], width, label="Hybrid")
plt.xticks(list(x), labels)
plt.ylim(0, 1.05)
plt.ylabel("Retention Rate")
plt.title("Average SOAP Retention by Mode")
plt.legend()
plt.tight_layout()
output_path = os.path.join(RESULTS_DIR, "avg_soap_retention.png")
plt.savefig(output_path, dpi=180)
print(f"✅ 生成圖表: {output_path}")
# plt.show()

# ============== 2) 臨床要素平均比較（長條圖） ==============
labels = elem_cols
x = range(len(labels))

plt.figure(figsize=(12, 6))
plt.bar([i - width for i in x], [avg_rule_e[c] for c in labels], width, label="Rule")
plt.bar([i           for i in x], [avg_sem_e[c]  for c in labels], width, label="Semantic")
plt.bar([i + width   for i in x], [avg_hyb_e[c]  for c in labels], width, label="Hybrid")
plt.xticks(list(x), labels, rotation=20, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("Recall Rate")
plt.title("Average Key-Element Recall by Mode")
plt.legend()
plt.tight_layout()
output_path = os.path.join(RESULTS_DIR, "avg_key_element_recall.png")
plt.savefig(output_path, dpi=180)
print(f"✅ 生成圖表: {output_path}")
# plt.show()

# ============== 3) 壓縮比比較（盒鬚圖） ==============
df_rule["_mode"] = "Rule"
df_sem["_mode"]  = "Semantic"
df_hyb["_mode"]  = "Hybrid"
df_all = pd.concat([df_rule, df_sem, df_hyb], ignore_index=True)

plt.figure(figsize=(10, 6))
df_all.boxplot(column="compression_ratio", by="_mode")
plt.suptitle("")  # 去掉預設 super title
plt.title("Compression Ratio by Mode")
plt.xlabel("Mode")
plt.ylabel("Compression Ratio")
plt.tight_layout()
output_path = os.path.join(RESULTS_DIR, "compression_ratio_box.png")
plt.savefig(output_path, dpi=180)
print(f"✅ 生成圖表: {output_path}")
# plt.show()

print(f"\n✅ 所有圖表已儲存至 {RESULTS_DIR}/")
