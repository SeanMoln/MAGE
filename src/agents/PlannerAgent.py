import openai
from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = os.getenv("openai_api_key")
class PlannerAgent:
    def __init__(self):
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def plan(self, task_description, input_text):
        plan_prompt = f"""
        Produce **exactly four action-oriented substeps** to guide prompt optimization
        for a concise SOAP-style clinical summary.

        Rules:
        - Output exactly 4 lines, each starting with "- ".
        - Each line must correspond to one SOAP section in order:
          1) Subjective → patient complaints, symptoms, history
          2) Objective → vitals, physical exam, key labs/imaging
          3) Assessment → diagnoses, impressions
          4) Plan → treatments, meds, follow-up
        - Substeps must be actions (e.g., "Summarize key symptoms"), not questions.
        - No headings, explanations, or extra text. Only 4 bullet lines.

        Task: {task_description}
        Input text: {input_text}
        """
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a clinical task planner for prompt optimization."},
                {"role": "user", "content": plan_prompt}
            ]
        )
        if response.usage:
            self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
            self.token_usage["completion_tokens"] += response.usage.completion_tokens
            self.token_usage["total_tokens"] += response.usage.total_tokens
        print("\n[Progress] Planning substeps...\n")
        return response.choices[0].message.content.strip().split("\n")

    def get_token_usage(self):
        return self.token_usage.copy()