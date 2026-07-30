# MARS / MAGE

**Multi-Agent Gap-detection Enrichment for Clinical Note Summarization**

以多代理缺漏補充架構（MAGE）從 MIMIC-III 出院摘要生成 SOAP 格式條列式摘要。

---

## 系統概覽

本專案實作兩條 Pipeline，均透過 `ManagerAgent` 執行：

| Pipeline | 方法 | 說明 |
|---|---|---|
| **Pipeline A：MARS（原始版）** | `run()` | Planner → Teacher-Critic-Student Loop × 4 → Target → PromptMemory |
| **Pipeline B：MAGE（缺漏補充版）** | `run_mars()` | Stage 1: PreDraft → Stage 2: TeacherGap-CriticGap-StudentGap Loop × 4 → TargetGap |

**核心創新（MAGE）**：Stage 1 以固定 SOAP 提示詞生成 PreDraft 初稿；Stage 2 以 PreDraft 為固定審查基準，透過缺漏偵測代理迴圈（最多 5 次回退）逐步補充遺漏關鍵臨床要素，最終由 TargetGap 生成補充後摘要。

---

## 完整執行流程

```
MIMIC-III 原始資料（preprocessing/data_raw/）
      │
      ▼
① python app.py preprocess
      │  data/preprocessing/discharge_preprocessed_YYYYMMDD_NNN.json
      ▼
② python app.py run-mage  （或 run-mars）
      │  data/mars/mage_YYYYMMDD_NNN.json
      ▼
③ python app.py evaluate
      │  data/evaluation/report_YYYYMMDD_NNN.json
      │  data/evaluation/metrics_YYYYMMDD_NNN.csv
      ▼
④ python evaluation/plot_eval_comparison.py
```

---

## 快速開始

### 環境設定

```bash
source .venv/bin/activate
pip install -r requirements.txt
echo "openai_api_key=your_key_here" > .env
```

### 資料來源：兩種方式

---

#### 方式一：MIMIC-III（預設）

需先取得 PhysioNet 授權，下載以下三個檔案至 `preprocessing/data_raw/`：

- `NOTEEVENTS.csv.gz`
- `ADMISSIONS.csv.gz`
- `DIAGNOSES_ICD.csv.gz`

取得後執行 `preprocess` 指令自動完成提取與清理。

---

#### 方式二：自訂資料集（不需 MIMIC-III）

`run-mars` 與 `run-mage` 的 `--input` 接受任何符合以下格式的 JSON 檔案，**不需要經過 `preprocess` 步驟**。

**格式：JSON 陣列，每個元素為一筆臨床文字記錄（字串）**

```json
[
  "Chief Complaint: Chest pain and shortness of breath.\n\nHistory of Present Illness:\nA 65-year-old male presented with...\n\nPhysical Exam:\nVitals: BP 140/90, HR 92...\n\nAssessment:\nAcute coronary syndrome...\n\nPlan:\nAdmit to CCU, start heparin drip...",
  "Patient is a 78-year-old female with history of CHF...",
  "..."
]
```

**準備步驟：**

1. 將你的臨床記錄整理為純文字字串
2. 存成 JSON 陣列格式（UTF-8 編碼）

```python
import json

notes = [
    "note 1 full text...",
    "note 2 full text...",
]

with open("data/preprocessing/my_dataset.json", "w", encoding="utf-8") as f:
    json.dump(notes, f, ensure_ascii=False, indent=2)
```

3. 直接以 `--input` 傳入 Pipeline：

```bash
python app.py run-mage --input data/preprocessing/my_dataset.json
python app.py run-mars --input data/preprocessing/my_dataset.json
```

**格式建議：**

- 建議包含結構化段落標題（如 `Chief Complaint:`、`Assessment:`、`Plan:` 等），有助於 SOAPParser 正確切割區段，進而提升摘要品質
- 每筆建議最少 150 詞，過短的記錄可能導致摘要過於簡略
- 文字不需預先清理（系統內部會進行基本清理）
- 語言：目前 Prompt 設計為英文臨床記錄；中文記錄理論上可用，但尚未針對中文臨床格式優化

---

## 指令說明

所有操作統一透過 `app.py` 執行。執行前須啟動虛擬環境：

```bash
source .venv/bin/activate
python app.py <command> [options]
```

---

### `preprocess` — 資料預處理

從 MIMIC-III 原始 CSV 提取出院摘要，進行清理與 SOAP 解析。

```bash
python app.py preprocess [--sample N] [--seed N] [--min-words N] [--icd-top-n N] [--output NAME]
```

| 參數 | 說明 | 預設 |
|---|---|---|
| `--sample N` | 抽取筆數 | 100 |
| `--seed N` | 隨機種子 | 42 |
| `--min-words N` | 最小字數門檻 | 150 |
| `--icd-top-n N` | ICD 碼取前 N 碼 | 5 |
| `--output NAME` | 輸出檔名（選填） | 自動生成 |

**Input**：`preprocessing/data_raw/`
**Output**：`data/preprocessing/discharge_preprocessed_YYYYMMDD_NNN.json`

```bash
# 標準執行
python app.py preprocess --sample 100 --seed 42

# 自訂輸出檔名
python app.py preprocess --sample 50 --output my_dataset.json
```

---

### `run-mage` — Pipeline B：MAGE 缺漏補充（主要）

以雙階段架構生成補充後 SOAP 摘要。

```bash
python app.py run-mage [--input FILE] [--output FILE] [--cosine N] [--jaccard N] [--limit N]
```

| 參數 | 說明 | 預設 |
|---|---|---|
| `--input FILE` | 預處理 JSON | 自動讀最新 `discharge_preprocessed_*.json` |
| `--output FILE` | 輸出路徑 | 自動生成 |
| `--cosine N` | 記憶庫 cosine 門檻 % | 92 |
| `--jaccard N` | 記憶庫 Jaccard 門檻 | 0.60 |
| `--limit N` | 只跑前 N 筆（0 = 全部） | 0 |

**Input**：`data/preprocessing/discharge_preprocessed_*.json`
**Output**：`data/mars/mage_YYYYMMDD_NNN.json`

```bash
# 全部筆數
python app.py run-mage

# 指定輸入，跑前 10 筆測試
python app.py run-mage --input data/preprocessing/discharge_preprocessed_20260319_001.json --limit 10
```

---

### `run-mars` — Pipeline A：MARS 原始

以 Teacher-Critic-Student 迴圈搭配 PromptMemory 優化提示詞後生成摘要。

```bash
python app.py run-mars [--input FILE] [--output FILE] [--cosine N] [--jaccard N] [--clear-memory] [--limit N]
```

| 參數 | 說明 | 預設 |
|---|---|---|
| `--input FILE` | 預處理 JSON | 自動讀最新 `discharge_preprocessed_*.json` |
| `--output FILE` | 輸出路徑 | 自動生成 |
| `--cosine N` | 記憶庫 cosine 門檻 % | 92 |
| `--jaccard N` | 記憶庫 Jaccard 門檻 | 0.60 |
| `--clear-memory` | 執行前清空 prompt memory | — |
| `--limit N` | 只跑前 N 筆（0 = 全部） | 0 |

**Input**：`data/preprocessing/discharge_preprocessed_*.json`
**Output**：`data/mars/results_YYYYMMDD_NNN.json`

```bash
python app.py run-mars --limit 5
python app.py run-mars --clear-memory --input data/preprocessing/discharge_preprocessed_20260319_001.json
```

---

### `evaluate` — 評估摘要品質

計算 SOAP 保留率與關鍵要素召回率。

```bash
python app.py evaluate --json FILE [--preprocessed FILE] [--mode MODE]
                       [--comp-limit N] [--lambda N]
                       [--theta-facet N] [--theta-element N]
                       [--tau-facet N] [--tau-element N]
                       [--report FILE] [--csv FILE]
```

| 參數 | 說明 | 預設 |
|---|---|---|
| `--json FILE` | MAGE/MARS 結果 JSON（**必填**） | — |
| `--preprocessed FILE` | 完整原文 JSON（評估更準確） | — |
| `--mode` | `hybrid` / `rule` / `semantic` | `hybrid` |
| `--comp-limit N` | 壓縮率上限 | 0.65 |
| `--lambda N` | λ rule 比重（hybrid 用） | 0.6 |
| `--report FILE` | 輸出報告路徑 | 自動生成 |
| `--csv FILE` | 輸出指標 CSV 路徑 | 自動生成 |

**Input**：`data/mars/mage_*.json` 或 `data/mars/results_*.json`
**Output**：`data/evaluation/report_YYYYMMDD_NNN.json` + `metrics_YYYYMMDD_NNN.csv`

```bash
# Hybrid 模式（完整評估，需 API）
python app.py evaluate \
  --json data/mars/mage_20260731_001.json \
  --preprocessed data/preprocessing/discharge_preprocessed_20260319_001.json \
  --mode hybrid

# Rule 模式（快速，不需 API）
python app.py evaluate --json data/mars/mage_20260731_001.json --mode rule
```

---

### 其他工具

**重置 Prompt 記憶庫：**

```bash
python -m src.utils.promtmemoryReset
```

**多系統統計比較（Wilcoxon signed-rank test）：**

```bash
python evaluation/compare_all_systems.py
```

**生成比較圖表：**

```bash
python evaluation/plot_eval_comparison.py
```

---

## 資料目錄

| 目錄 | 檔名格式 | 內容 |
|---|---|---|
| `data/preprocessing/` | `discharge_preprocessed_YYYYMMDD_NNN.json` | 預處理後的臨床文字陣列，每筆為一個字串，可直接作為 Pipeline 輸入 |
| `data/mars/` | `mage_YYYYMMDD_NNN.json` / `results_YYYYMMDD_NNN.json` | Pipeline 執行結果，含原文、PreDraft、最終摘要與 token 統計 |
| `data/evaluation/` | `report_YYYYMMDD_NNN.json` / `metrics_YYYYMMDD_NNN.csv` | 各案例的 SOAP 保留率、關鍵要素召回率與壓縮比 |
| `data/memory/` | `prompt_memory.json` | PromptMemoryAgent 的語義記憶庫（Pipeline A 使用） |

---

## 專案結構

```
MARS_PMB/
├── app.py                          # 統一 CLI 入口（preprocess / run-mars / run-mage / evaluate）
├── src/agents/
│   ├── ManagerAgent.py             # 主控制器（run() / run_mars()）
│   ├── PlannerAgent.py             # SOAP 步驟規劃
│   ├── TeacherAgent.py             # Socratic 提問（Pipeline A）
│   ├── CriticAgent.py              # 問題驗證（Pipeline A）
│   ├── StudentAgent.py             # 提示詞優化（Pipeline A）
│   ├── TargetAgent.py              # SOAP 摘要生成（Pipeline A / Stage 1）
│   ├── TeacherAgentGap.py          # 缺漏偵測提問（Pipeline B Stage 2）
│   ├── CriticAgentGap.py           # 缺漏問題驗證（Pipeline B Stage 2）
│   ├── StudentAgentGap.py          # 缺漏補充指令累積（Pipeline B Stage 2）
│   ├── TargetAgentGap.py           # 補充後摘要生成（Pipeline B Stage 2）
│   └── PromptMemoryAgent.py        # 語義記憶庫（Pipeline A）
│
├── preprocessing/
│   ├── src/
│   │   ├── extract_discharge.py    # 主要預處理腳本
│   │   ├── soap_parser.py              # SOAP 區段解析（hybrid 模式）
│   │   └── extract_from_local.py
│   └── config/data_config.yaml
│
└── evaluation/
    ├── evalute_summary.py              # 混合評分引擎
    ├── compare_all_systems.py          # 多系統統計比較
    └── plot_eval_comparison.py         # 視覺化
```

---

## MAGE 架構細節

### 代理組成

| 代理 | 階段 | 職責 |
|---|---|---|
| **TargetAgent** | Stage 1 | 以固定 SOAP 提示詞生成條列式 PreDraft 初稿 |
| **TeacherAgentGap** | Stage 2 | 以 PreDraft 為審查基準，提出缺漏偵測問題 |
| **CriticAgentGap** | Stage 2 | 驗證問題是否聚焦於缺漏（拒絕壓縮類問題）；未通過則回退理由給 Teacher，最多重試 5 次 |
| **StudentAgentGap** | Stage 2 | 依合格問題累積補充指令至 current_prompt |
| **TargetAgentGap** | Stage 2 | 依補充提示詞與 PreDraft 生成最終摘要 |

### Stage 2 缺漏偵測迴圈

```
for each SOAP substep in [S, O, A, P]:
    for attempt in range(5):
        TeacherGap → 提出缺漏問題（若有 Critic 回退理由則一併傳入）
        CriticGap  → 驗證問題
        if 通過 → break
        else    → 回退理由傳回 Teacher，下一輪修正
    StudentGap → 依通過的問題補充 current_prompt
TargetGap → 生成最終補充後摘要
```

### 核心設計決策

**1. PreDraft 固定基準**
TeacherGap 每輪審查的 PreDraft 固定不變（不隨迭代演化），確保每步都審查同一份初稿。

**2. 雙門檻記憶體檢索（Pipeline A）**
- Cosine similarity ≥ 92%（`text-embedding-3-small`）
- Jaccard 實體重疊 ≥ 0.60
- 兩條件同時達到 → 跳過完整優化，執行 2 步快速精煉

**3. TargetAgentGap 動態詞數上限**
```
max_words = max(PreDraft 詞數 + 50, 250, 原文詞數 × 0.65)
```

---

## 評估方法

### SOAP 保留率（Hybrid）

```
H = max(R_rule, R_semantic)
```

### 關鍵要素召回率

追蹤 6 類臨床要素：`diagnoses`、`labs`、`imaging`、`comorbidities`、`plan_general`、`plan_minimums_adhf`

NA 處理：原文中不存在的要素標記 N/A，不計入巨集平均。

### 壓縮比

- 上限：≤ 0.65（摘要詞數 / 原文詞數）
- 以 results JSON 的 `input_length` 欄位計算（完整詞數，非截斷的 `input_text`）

---

## 使用模型

| 代理 | 模型 |
|---|---|
| Planner、Teacher/Gap、Student/Gap、Target/Gap | `gpt-4o-mini` |
| CriticAgent（Pipeline A） | `gpt-4` |
| CriticAgentGap（Pipeline B） | `gpt-4o-mini` |
| PromptMemoryAgent 語義嵌入 | `text-embedding-3-small` |

---

## 安全性提醒

- 不要將 `.env` 提交到版本控制
- MIMIC-III 資料需 PhysioNet 認證（CITI 訓練 + DUA 簽署）

---

**Python 版本**：3.12+
**主要依賴**：`openai`、`scikit-learn`、`pandas`、`tqdm`、`python-dotenv`
