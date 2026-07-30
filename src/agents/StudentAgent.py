import openai

import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("openai_api_key")
class StudentAgent:
    def __init__(self):
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def improve_prompt(self, question, current_prompt):
        student_prompt = f"""
        Refine the current prompt using the Teacher's question to improve precision,
        compression, and clinical focus for a concise SOAP-style summary.

        Refinement rules:
        - Enforce SOAP structure: Subjective, Objective, Assessment, Plan.
        - Subjective: ≤2 key symptoms/complaints, combine overlapping symptoms.
        - Objective: ≤3–4 items, include at least one abnormal lab or imaging if present.
        - Assessment: main diagnosis + ≤1 differential, ≤1 line.
        - Plan: ≤3–4 concise actionable steps (meds, consult, follow-up).
        - Use only information explicitly in the note; mark missing details as "unspecified".
        - Each section ≤1–2 lines; avoid full sentences or narrative style.

        Current Prompt: {current_prompt}
        Teacher's Question: {question}

        Output only the new improved prompt.
        """
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a clinical prompt engineer. Refine prompts to produce concise, accurate SOAP-style summaries using only information present in the clinical note."},
                {"role": "user", "content": student_prompt}
            ]
        )
        if response.usage:
            self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
            self.token_usage["completion_tokens"] += response.usage.completion_tokens
            self.token_usage["total_tokens"] += response.usage.total_tokens
        print("\n[Progress] Improving prompt based on Socratic question...\n")
        if not response.choices:
            return "No response from model."
        return response.choices[0].message.content.strip()

    def get_token_usage(self):
        return self.token_usage.copy()