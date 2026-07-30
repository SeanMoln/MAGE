# PromptMemoryAgent.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import pandas as pd
import os
import re
import time
import openai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
openai.api_key = os.getenv("openai_api_key")

class PromptMemoryAgent:
    def __init__(self, json_path=None, excel_path=None):
        # 如果沒有提供路徑，使用專案根目錄的絕對路徑
        if json_path is None:
            # 找到專案根目錄（假設此檔案在 src/agents/）
            project_root = Path(__file__).parent.parent.parent
            json_path = project_root / "data" / "memory" / "prompt_memory.json"
        if excel_path is None:
            project_root = Path(__file__).parent.parent.parent
            excel_path = project_root / "data" / "memory" / "prompt_memory.xlsx"

        self.json_path = str(json_path)
        self.excel_path = str(excel_path)
        self.memory = []

        # 確保目錄存在
        Path(self.json_path).parent.mkdir(parents=True, exist_ok=True)

        self._load()

    # ---------- 基礎 I/O ----------
    def _load(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                try:
                    self.memory = json.load(f)
                except Exception:
                    self.memory = []
        else:
            self.memory = []

    def save(self):
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)
        try:
            df = pd.DataFrame(self.memory)
            df.to_excel(self.excel_path, index=False)
        except Exception:
            pass  # 沒安裝 xlsx 相依可忽略

    # ---------- Embedding / 簡易實體 ----------
    def _embed(self, text: str):
        # text-embedding-3-small 上限 8192 tokens，以字元數粗略截斷（約 2.5-3 字元/token，保守取 2.5）
        MAX_CHARS = 8192 * 2
        text = (text or "")[:MAX_CHARS]
        resp = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text if text else ""]
        )
        return resp.data[0].embedding

    def _extract_entities(self, text: str):
        meds = re.findall(r'\b(aspirin|metformin|furosemide|lisinopril|insulin)\b', text, flags=re.I)
        vitals = re.findall(r'(BP|blood pressure|HR|heart rate|RR|respiratory rate|Temp|temperature|SpO2|oxygen saturation)', text, flags=re.I)
        labs = re.findall(r'\b(BNP|CRP|creatinine|hCG|Na|K|HbA1c)\b', text, flags=re.I)
        dx = re.findall(r'\b(heart failure|pneumonia|appendicitis|ectopic pregnancy|diabetes)\b', text, flags=re.I)
        return set(map(str.lower, meds + vitals + labs + dx))

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        return len(a & b) / max(1, len(a | b))

    # ---------- 寫入記憶（以病例文本建向量） ----------
    def add_record(self,
                   task_description: str,
                   optimized_prompt: str,
                   result: str,
                   input_text: str = None,
                   doc_type: str = "SOAP",
                   department: str = None,
                   quality_score: float = None,
                   prompt_version: str = "v1"):
        text_for_embed = input_text if input_text else task_description
        embedding = self._embed(text_for_embed)
        entities = list(self._extract_entities(input_text or ""))

        record = {
            "ts": int(time.time()),
            "doc_type": doc_type,
            "department": department,
            "task_description": task_description,
            "optimized_prompt": optimized_prompt,
            "result": result,
            "embedding": embedding,
            "entities": entities,
            "quality_score": quality_score,
            "prompt_version": prompt_version,
            "input_len": len((input_text or "").split())
        }
        self.memory.append(record)
        self.save()

    # ---------- 新查詢：用病例文本檢索 ----------
    def query_similar_cases(self,
                            input_text: str,
                            doc_type: str = "SOAP",
                            department: str = None,
                            top_k: int = 3,
                            min_quality: float = 0.0,
                            max_age_days: int = 365):
        if not self.memory:
            return []

        input_vec = self._embed(input_text or "")
        matrix = np.array([m["embedding"] for m in self.memory if "embedding" in m])
        if matrix.size == 0:
            return []

        sims = cosine_similarity([input_vec], matrix)[0]

        now = int(time.time())
        candidates = []
        for i, m in enumerate(self.memory):
            if doc_type and m.get("doc_type") and m["doc_type"] != doc_type:
                continue
            if department and m.get("department") and m["department"] != department:
                continue
            age_days = (now - m.get("ts", now)) / (60 * 60 * 24)
            if age_days > max_age_days:
                continue
            q = m.get("quality_score")
            if q is not None and q < min_quality:
                continue

            sim = float(sims[i]) if i < len(sims) else 0.0
            item = dict(m)
            item["similarity"] = round(sim * 100.0, 2)
            candidates.append(item)

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:top_k]

    # ---------- 舊 API 相容 ----------
    def query_similar_tasks(self, text_or_task: str, top_k: int = 2, return_score: bool = True):
        if not self.memory:
            return []
        input_vec = self._embed(text_or_task or "")
        matrix = np.array([m["embedding"] for m in self.memory if "embedding" in m])
        if matrix.size == 0:
            return []
        sims = cosine_similarity([input_vec], matrix)[0]
        top_indices = np.argsort(sims)[::-1][:max(1, top_k)]
        results = []
        for i in top_indices:
            item = dict(self.memory[i])
            if return_score:
                item["similarity"] = round(float(sims[i]) * 100.0, 2)
            results.append(item)
        return results
