import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("openai_api_key")


class CriticAgentGap:
    """
    Gap-detection 版 CriticAgent。
    驗證 TeacherAgentGap 產生的問題是否真正聚焦在「缺漏偵測」，
    而非壓縮、優先順序或格式。
    """

    def __init__(self):
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def judge_question(self, question: str, baseline_draft: str = "") -> str:
        baseline_section = ""
        if baseline_draft:
            baseline_section = f"""
Baseline Draft (what the current summary already contains):
\"\"\"
{baseline_draft}
\"\"\"

"""
        critic_prompt = f"""
Judge whether the following Teacher question is suitable for **gap detection** in a clinical draft summary.

{baseline_section}Evaluation criteria:
1. Completeness focus:
   - The question must ask about information that is **missing, absent, or under-represented** in the draft.
   - It must NOT be about compression, brevity, prioritization, or selecting top-N items.
2. Grounded in the baseline (if provided):
   - The gap the question points to must be genuinely absent or insufficiently documented in the Baseline Draft above.
   - Reject questions that ask about information already clearly present in the baseline.
3. Socratic style:
   - Open-ended and probing — does not provide answers or state facts directly.
4. Clinical SOAP relevance:
   - Clearly tied to one SOAP section (S/O/A/P).
   - Asks about concrete clinical elements (lab values, medications, diagnoses, symptoms, imaging findings, follow-up items, etc.).
5. Prompt-actionable:
   - A prompt engineer must be able to act on this question by adding specific instructions to include missing information.

Output format:
- If valid:   [True]
- If invalid: [False] [suggestion: <reason why it fails and how to fix>]

Question:
{question}
"""
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict quality reviewer for gap-detection questions. "
                        "Reject any question that focuses on compression, brevity, or prioritization."
                    ),
                },
                {"role": "user", "content": critic_prompt},
            ],
        )
        if response.usage:
            self.token_usage["prompt_tokens"]     += response.usage.prompt_tokens
            self.token_usage["completion_tokens"] += response.usage.completion_tokens
            self.token_usage["total_tokens"]      += response.usage.total_tokens
        return response.choices[0].message.content.strip()

    def get_token_usage(self):
        return self.token_usage.copy()
