# ManagerAgent.py
from . import UserProxyAgent
from . import PlannerAgent
from . import TeacherAgent
from . import CriticAgent
from . import StudentAgent
from . import TargetAgent
from . import PromptMemoryAgent
from .TeacherAgentGap import TeacherAgentGap
from .CriticAgentGap import CriticAgentGap
from .StudentAgentGap import StudentAgentGap
from .TargetAgentGap import TargetAgentGap
from tqdm import tqdm
import re

# Pre_Draft 固定 prompt：Stage 1 初稿生成（作為 MARS Stage 2 的審查基準）
_PRE_DRAFT_PROMPT = """Generate a complete and accurate clinical summary in SOAP format.

Structure:
- **Subjective**: All key symptoms and complaints explicitly mentioned
- **Objective**: All vital signs, exam findings, lab results with values, imaging findings
- **Assessment**: All diagnoses including inpatient complications
- **Plan**: All medications (with doses if stated), consultations, procedures, follow-up instructions, return-to-ED criteria

Preserve ALL explicitly documented clinical information. Do not fabricate or infer values not present in the note."""

class ManagerAgent:

    @staticmethod
    def _critic_passed(judge: str) -> bool:
        """
        Critic 通過判定，處理兩種輸出格式：
          1. 單一回應：以 [True] 開頭
          2. 逐題回應：每題各有 [True]/[False]，全部為 True 才算通過
        """
        j = judge.strip()
        if j.startswith("[True]"):
            return True
        verdicts = re.findall(r'\[(True|False)\]', j, re.IGNORECASE)
        return bool(verdicts) and all(v.lower() == "true" for v in verdicts)

    # -執行序-
    def __init__(self):
        self.user_proxy = UserProxyAgent.UserProxyAgent()
        self.planner = PlannerAgent.PlannerAgent()
        self.teacher = TeacherAgent.TeacherAgent()
        self.critic = CriticAgent.CriticAgent()
        self.student = StudentAgent.StudentAgent()
        self.target = TargetAgent.TargetAgent()
        self.memory = PromptMemoryAgent.PromptMemoryAgent()
        # Pre_Draft 專用 TargetAgent（Stage 1，與 MARS Stage 2 的 target 分開計算 token）
        self.pre_draft_target = TargetAgent.TargetAgent()
        # MARS Stage 2 agents（Gap-detection 優化）
        self.teacher_gap = TeacherAgentGap()
        self.critic_gap  = CriticAgentGap()
        self.student_gap = StudentAgentGap()
        self.target_gap  = TargetAgentGap()

    # --- 將規劃輸出正規化為 S/O/A/P 4 步 ---
    def _normalize_to_soap_steps(self, raw_text_or_list, word_count: int):
        """
        將 Planner 回傳內容正規化為 Exactly 4 個、依序 S/O/A/P 的動作型步驟。
        若模型回的是前處理類步驟（移除行政資訊/突出最近事件），在一般長度下會忽略；
        超長文本 (>1000 詞) 則把前處理融合進 S 的開頭敘述。
        """
        # 轉成行
        if isinstance(raw_text_or_list, str):
            lines = raw_text_or_list.splitlines()
        elif isinstance(raw_text_or_list, list):
            lines = [str(x) for x in raw_text_or_list]
        else:
            lines = []

        # 僅取以 "- " 開頭的動作行
        bullets = []
        for line in lines:
            s = line.strip()
            if s.startswith("- "):
                bullets.append(s[2:].strip())

        # 關鍵字對應（寬鬆比對）
        def is_subjective(s):  return bool(re.search(r"\bsubjective\b|complaint|symptom|history", s, re.I))
        def is_objective(s):   return bool(re.search(r"\bobjective\b|vital|exam|lab|imaging|test result", s, re.I))
        def is_assessment(s):  return bool(re.search(r"\bassessment\b|diagnos|impression|reasoning|differential", s, re.I))
        def is_plan(s):        return bool(re.search(r"\bplan\b|treatment|medication|follow[- ]?up|discharge", s, re.I))
        def is_preprocess(s):  return bool(re.search(r"administrative|non-?clinical|remove|filter|recent events|chronolog", s, re.I))

        subj, obj, assess, plan = [], [], [], []
        for b in bullets:
            if is_subjective(b): subj.append(b)
            elif is_objective(b): obj.append(b)
            elif is_assessment(b): assess.append(b)
            elif is_plan(b): plan.append(b)
            elif is_preprocess(b): 
                # 一般情況忽略；超長文本時後續合併進 S
                pass

        # 預設動作（防空）
        defaults = [
            "Identify key patient-reported complaints and symptoms (Subjective)",
            "Extract vital signs, physical exam findings, labs, and imaging (Objective)",
            "Summarize main diagnoses and clinical reasoning (Assessment)",
            "Condense treatments, medications, and follow-up (Plan)",
        ]

        S = (subj[0] if subj else defaults[0])
        O = (obj[0]  if obj  else defaults[1])
        A = (assess[0] if assess else defaults[2])
        P = (plan[0] if plan else defaults[3])

        steps = [S, O, A, P]

        # 超長文本：在 S 前半加入前處理描述（仍維持 4 步）
        if word_count > 1000:
            pre = "Pre-filter non-clinical admin details and prioritize most recent clinically relevant events"
            steps[0] = f"{pre}; then {steps[0]}"

        return steps

    def run(self, task_description, input_text, doc_type="SOAP", department=None):
        input_data = self.user_proxy.receive_input(task_description, input_text)
        base_prompt = input_data['base_prompt']

        word_count = len(input_text.split())
        print("input_text length:", word_count, "words")

        # --- 超短文本：直接摘要 ---
        if word_count < 50:
            print("[Mode] Simple Mode: skipping full Socratic loop.")
            current_prompt = base_prompt

        else:
            # --- 以病例文本檢索相似案例（不再用 task_description） ---
            similar_cases = self.memory.query_similar_cases(
                input_text=input_text,
                doc_type=doc_type,
                department=department,
                top_k=3,
                min_quality=0.0,
                max_age_days=365
            )

            # --- 雙門檻：文本相似 + 簡易實體重疊 ---
            current_prompt = base_prompt
            is_high_sim = False
            ENT_JACCARD_MIN = 0.60
            SIM_THRESHOLD = 92.0

            if similar_cases:
                best = similar_cases[0]
                ent_input = self.memory._extract_entities(input_text)
                ent_mem = set(map(str.lower, best.get("entities", [])))
                jacc = self.memory._jaccard(ent_input, ent_mem)

                if best["similarity"] >= SIM_THRESHOLD and jacc >= ENT_JACCARD_MIN:
                    print("\n[Memory] High-similarity case found. Using prior prompt + quick refinement.\n")
                    current_prompt = best["optimized_prompt"]
                    is_high_sim = True

            # --- 規劃與微調：正規化為 S/O/A/P；高相似跑 2 步，否則跑 4 步 ---
            raw = self.planner.plan(task_description, input_data["input_text"])
            steps = self._normalize_to_soap_steps(raw, word_count)

            max_steps = 2 if is_high_sim else 4
            steps = steps[:min(max_steps, len(steps))]

            print(f"\n[Progress] The number of steps: {len(steps)}\n")
            print("\n[Progress] Optimizing prompt over", len(steps), "steps:\n")

            for step in tqdm(steps, desc="Optimizing", ncols=80):
                print("\n[Progress] step:", step)
                question = None
                critic_feedback = ""
                for _ in range(5):
                    q_try = self.teacher.ask_questions(step, current_prompt, feedback=critic_feedback)
                    judge = self.critic.judge_question(q_try)
                    if self._critic_passed(judge):
                        question = q_try
                        break
                    m = re.search(r'\[suggestion:\s*(.*?)\]', judge, re.DOTALL | re.IGNORECASE)
                    critic_feedback = m.group(1).strip() if m else judge
                    print(f"\n[Progress] ⚠️ Critic 未通過，回退理由：{critic_feedback}")
                if question is None:
                    question = f"How can the prompt better emphasize this step: {step}?"
                    print("[Progress] 保底問題啟用")

                current_prompt = self.student.improve_prompt(question, current_prompt)

        # --- 生成最終輸出 ---
       
        result = self.target.evaluate(current_prompt, input_text)

        # --- 回寫記憶（含病例文本與中介資料欄位；科別可選填） ---
        self.memory.add_record(
            task_description=task_description,
            optimized_prompt=current_prompt,
            result=result,
            input_text=input_text,
            doc_type=doc_type,
            department=department,
            quality_score=None,
            prompt_version="v1"
        )

        return current_prompt, result

    def run_mars(self, task_description, input_text, doc_type="SOAP", department=None):
        """
        MARS 兩階段 pipeline：
          Stage 1 — Pre_Draft：TargetAgent 以固定 prompt 生成初稿（作為審查基準）
          Stage 2 — Gap-detection loop（TeacherGap/CriticGap/StudentGap）
                    針對 Pre_Draft 逐 SOAP 段詢問「缺漏哪些資訊」，補充後再跑 TargetAgentGap

        回傳：(optimized_prompt, final_result, pre_draft_result)
        """
        # ── Stage 1：Pre_Draft 初稿 ─────────────────────────────────────────
        print("\n[Stage 1] Pre_Draft：TargetAgent 生成初稿…\n")
        pre_draft_result = self.pre_draft_target.evaluate(_PRE_DRAFT_PROMPT, input_text)
        print(f"[Stage 1] Pre_Draft 長度: {len(pre_draft_result.split())} 詞")
        print("[Stage 1] Pre_Draft 預覽:\n", pre_draft_result[:300], "…\n")

        # ── Stage 2：Gap-detection 優化 ─────────────────────────────────────
        print("[Stage 2] Gap-detection：針對 Pre_Draft 逐段詢問缺漏資訊…\n")

        # Stage 2 固定使用 SOAP 四段作為缺漏偵測框架
        steps = [
            (
                "Subjective — Detect missing elements: chief complaint, symptom onset/duration/severity, "
                "associated symptoms, aggravating/relieving factors, relevant past medical history, "
                "prior hospitalizations, current medications, allergies, and social/family history."
            ),
            (
                "Objective — Detect missing elements: all vital signs with numeric values (BP, HR, RR, Temp, SpO2), "
                "physical examination findings, all laboratory results with values and units, "
                "imaging findings (X-ray, CT, MRI, ultrasound), and microbiology/culture results."
            ),
            (
                "Assessment — Detect missing elements: primary diagnosis, secondary diagnoses, "
                "all active comorbidities, problem list, and clinical reasoning or differential diagnoses considered."
            ),
            (
                "Plan — Detect missing elements: all medications with dose/route/frequency, "
                "consultations ordered or obtained, specific follow-up instructions (date, location, provider), "
                "return-to-ED criteria, pending lab or imaging results, and discharge condition/disposition."
            ),
        ]
        print(f"[Stage 2] SOAP 步驟: {steps}\n")

        # current_prompt 從固定 Pre_Draft 指令出發（不是初稿文字）
        # pre_draft_result 作為固定參照，每輪都傳給 TeacherGap 與 StudentGap
        current_prompt = _PRE_DRAFT_PROMPT

        for step in tqdm(steps, desc="MARS Optimizing", ncols=80):
            print(f"\n[Stage 2] 步驟: {step}")

            question = None
            critic_feedback = ""
            for _ in range(5):
                q_try = self.teacher_gap.ask_questions(step, pre_draft_result, feedback=critic_feedback)
                judge = self.critic_gap.judge_question(q_try, pre_draft_result)
                if self._critic_passed(judge):
                    question = q_try
                    break
                m = re.search(r'\[suggestion:\s*(.*?)\]', judge, re.DOTALL | re.IGNORECASE)
                critic_feedback = m.group(1).strip() if m else judge
                print(f"[Stage 2] ⚠️ Critic 未通過，回退理由：{critic_feedback}")

            if question is None:
                question = (
                    f"What clinical information related to '{step}' "
                    f"is absent or insufficiently detailed in this pre-draft summary?"
                )
                print("[Stage 2] 保底 gap 問題啟用")

            print(f"[Stage 2] Gap 問題:\n{question}\n")
            # StudentGap 同時知道「現有指令」和「Pre_Draft 已有的內容」
            current_prompt = self.student_gap.improve_prompt(
                question, current_prompt, pre_draft_result
            )

        # ── 最終摘要：gap-optimized prompt + Pre_Draft 參照 + 原始病歷 ────────
        print("\n[Stage 2] Gap-optimized prompt → TargetAgentGap 生成完整 MARS 摘要…\n")
        final_result = self.target_gap.evaluate(current_prompt, input_text, pre_draft_result)

        # MARS pipeline 不寫入 PMB：
        # gap-filling prompt 是針對單一 pre_draft 的補充指令，語意上不適合作為 SOAP 記憶庫的 seed

        return current_prompt, final_result, pre_draft_result

    # 向下相容別名（舊程式碼使用 run_with_baseline）
    def run_with_baseline(self, task_description, input_text, doc_type="SOAP", department=None):
        return self.run_mars(task_description, input_text, doc_type, department)

    def get_token_usage(self):
        """取得所有 agent 的累計 token 使用量（含 MARS agents 與 pre_draft_target）"""
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for agent in [self.planner, self.teacher, self.critic, self.student,
                      self.target, self.pre_draft_target,
                      self.teacher_gap, self.critic_gap, self.student_gap, self.target_gap]:
            if hasattr(agent, "get_token_usage"):
                for k, v in agent.get_token_usage().items():
                    total[k] += v
        return total

import json
from pathlib import Path
# --執行--
if __name__ == "__main__":
    task = """Summarize the following clinical note into a concise SOAP-style summary."""
    
    json_path = Path("../../data/input/soap_cases.json")
    with json_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    results = []
    for idx, input_text in enumerate(cases):
        print(f"\n===== Case {idx} =====")
        manager = ManagerAgent()
        memory = PromptMemoryAgent.PromptMemoryAgent()

        optimized_prompt, final_result = manager.run(task, input_text)
        print("\n[Final Optimized Prompt]\n", optimized_prompt)
        print("\n[Final Result]\n", final_result)
        similar = memory.query_similar_cases(input_text, top_k=2)
        case_result = {

            "input_text": input_text,
            "optimized_prompt": optimized_prompt,
            "final_result": final_result
            # "similar_cases": similar
        }
        results.append(case_result)
    output_path = Path("../../data/output/results.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All results saved to {output_path}")
        # for i, item in enumerate(similar):
        #     print(f"\nMatch {i+1} (Similarity: {item['similarity']}%):")
        #     print("DocType:", item.get("doc_type"))
        #     print("Prompt:", item.get("optimized_prompt", "")[:60])
        #     print("Result:", item.get("result", "")[:60])
    # input_text = content
    # manager = ManagerAgent()
    # optimized_prompt, final_result = manager.run(task, input_text)
    # print("\n[Final Optimized Prompt]\n", optimized_prompt)
    # print("\n[Final Result]\n", final_result)

    # # 示範：查相似案例（以病例文本檢索）
    # memory = PromptMemoryAgent.PromptMemoryAgent()
    # similar = memory.query_similar_cases(input_text, top_k=2)
    # for i, item in enumerate(similar):
    #     print(f"\nMatch {i+1} (Similarity: {item['similarity']}%):")
    #     print("DocType:", item.get("doc_type"))
    #     print("Prompt:", item.get("optimized_prompt", "")[:60])
    #     print("Result:", item.get("result", "")[:60])
