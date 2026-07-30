import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("openai_api_key")


class TeacherAgentGap:
    """
    Gap-detection 版 TeacherAgent。
    專門針對「Baseline 初稿」提問：
      - 該 SOAP 段落遺漏了哪些臨床資訊？
      - 哪些細節未被補充進摘要？
    不鼓勵壓縮，不提「prioritize」或「brevity」。
    """

    def __init__(self):
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def ask_questions(self, substep: str, baseline_draft: str, feedback: str = "") -> str:
        """
        Parameters
        ----------
        substep        : 目前 SOAP 段落的優化目標（由 PlannerAgent 產生）
        baseline_draft : Baseline TargetAgent 的初稿摘要（固定參照，不隨迭代改變）
        feedback       : Critic 上次拒絕的理由（重試時傳入，空字串代表首次嘗試）
        """
        feedback_section = ""
        if feedback:
            feedback_section = f"""
⚠️ Your previous questions were rejected by the Critic for the following reason:
\"{feedback}\"
Please revise your questions to directly address this feedback before submitting again.

"""
        prompt = f"""
You are an experienced clinical physician reviewing a baseline draft summary of a patient's clinical record.

Premise:
You know precisely what a **complete, standard clinical document** should contain in each SOAP section.
Using that clinical standard as your reference, read the baseline draft below and identify what is
**missing, absent, or insufficiently documented** compared to what a proper clinical record would normally include.

Your task:
Ask **2 open-ended Socratic questions** that guide a prompt engineer to detect and fill **information gaps**
in the baseline draft for the following SOAP subtask.

Constraints:
- Ground your questions in **standard clinical documentation norms** —
  ask what a physician would expect to see but cannot find in this draft.
- Focus solely on **completeness**: what is absent or under-documented.
- Questions must be specific to the SOAP section and clinically grounded
  (e.g., missing lab values with units, unnamed medications, undocumented vitals,
   absent comorbidities, incomplete follow-up instructions).
- Do NOT ask about style, formatting, sentence length, or structure.
- Do NOT suggest answers — only ask probing questions.
- Do NOT use words like "concise", "prioritize", "brevity", "limit", or "top N".

{feedback_section}SOAP Subtask: {substep}

Baseline Draft Summary (the draft being audited for gaps):
\"\"\"
{baseline_draft}
\"\"\"
"""
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an experienced clinical physician. "
                        "You know exactly what information belongs in each section of a standard clinical document. "
                        "Your job is to identify what is MISSING from a draft summary by comparing it against "
                        "standard clinical documentation norms — ask probing questions, never about compression."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        if response.usage:
            self.token_usage["prompt_tokens"]     += response.usage.prompt_tokens
            self.token_usage["completion_tokens"] += response.usage.completion_tokens
            self.token_usage["total_tokens"]      += response.usage.total_tokens
        print("\n[Gap Teacher] 詢問缺漏資訊...\n")
        if not response.choices:
            return "No response from model."
        return response.choices[0].message.content.strip()

    def get_token_usage(self):
        return self.token_usage.copy()
