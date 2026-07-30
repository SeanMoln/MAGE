import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("openai_api_key")


class StudentAgentGap:
    """
    Gap-filling 版 StudentAgent。
    根據 TeacherAgentGap 的缺漏問題，改寫 prompt 以明確要求
    補充原稿遺漏的臨床資訊——不引入壓縮語言。

    禁止規則（繼承自原 StudentAgent 修正版，並強化）：
    1. NEVER write "select top N" / "most critical N" / "2-3 prioritized"
    2. NEVER reference Teacher/Student/Critic/agent names in output
    3. NEVER include meta-instructions like "Address the Teacher's questions"
    4. NEVER use "for brevity", "concise", "limit to", "avoid excess detail"
    5. NEVER restrict any category to a fixed count
    """

    def __init__(self):
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def improve_prompt(self, question: str, current_prompt: str, baseline_draft: str) -> str:
        """
        Parameters
        ----------
        question       : TeacherAgentGap 產生的缺漏偵測問題
        current_prompt : 目前累積的指令 prompt（從 _BASELINE_PROMPT 開始逐步精煉）
        baseline_draft : Baseline TargetAgent 的初稿摘要（用來對照「已有什麼、缺什麼」）
        """
        student_prompt = f"""
You are a clinical prompt engineer. Your task is to refine a summarization prompt so that
it explicitly requires the clinical information that is **missing from the baseline draft**.

Context:
- The "Baseline Draft" below shows what the current summarization prompt already produces.
- The "Gap-Detection Question" identifies what clinical information is absent from that draft.
- The "Current Prompt" is the instruction prompt to be refined.

Your task:
Rewrite the Current Prompt to explicitly add instructions that require the missing information
identified by the Gap-Detection Question — information that is absent from the Baseline Draft.

Refinement rules:
- Preserve all existing instructions in the Current Prompt; only ADD what is missing.
- For any clinical category mentioned in the gap question, add an explicit INCLUDE ALL instruction.
- Do NOT add compression instructions. Do NOT say "limit to", "top N", "most important N",
  "concise", "brief", or "for brevity".
- Do NOT reference Teacher, Student, Critic, or any agent name in the output.
- Do NOT write meta-instructions like "Address the gap question by...".
- Write only the refined prompt text itself — no explanations, no preamble.

Baseline Draft (what the current prompt already produces — use this to identify what is already present):
\"\"\"
{baseline_draft}
\"\"\"

Gap-Detection Question (what is missing from the baseline draft):
{question}

Current Prompt (to be refined):
{current_prompt}

Output only the new improved prompt.
"""
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical prompt engineer specializing in completeness. "
                        "Your goal is to make prompts that capture ALL relevant clinical information — "
                        "never compress, never limit counts, never omit."
                    ),
                },
                {"role": "user", "content": student_prompt},
            ],
        )
        if response.usage:
            self.token_usage["prompt_tokens"]     += response.usage.prompt_tokens
            self.token_usage["completion_tokens"] += response.usage.completion_tokens
            self.token_usage["total_tokens"]      += response.usage.total_tokens
        print("\n[Gap Student] 依缺漏問題補充 prompt...\n")
        if not response.choices:
            return "No response from model."
        return response.choices[0].message.content.strip()

    def get_token_usage(self):
        return self.token_usage.copy()
