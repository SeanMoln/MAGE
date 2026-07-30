import openai
from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = os.getenv("openai_api_key")
class CriticAgent:
    def __init__(self):
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def judge_question(self, question):
        critic_prompt =f"""
        Judge whether the following Teacher question meets both **Socratic principles** and **clinical SOAP relevance**.

        Evaluation criteria:
        1. Socratic style:
        - Open-ended, probing, guiding thought.
        - Should not provide answers or state facts directly.
        2. Clinical SOAP relevance:
        - Must align with one SOAP section (S/O/A/P).
        - Must focus on clinical elements (symptoms, exam findings, diagnoses, treatments).
        - Must not encourage invention or inference beyond the input text.
        3. Clarity:
        - Must be clear, specific, and useful for improving a SOAP summarization prompt.

        Output format:
        - If valid: [True]
        - If invalid: [False] [suggestion: <reason>]

        Question: {question}
        """
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Socratic style critic."},
                {"role": "user", "content": critic_prompt}
            ]
        )
        if response.usage:
            self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
            self.token_usage["completion_tokens"] += response.usage.completion_tokens
            self.token_usage["total_tokens"] += response.usage.total_tokens
        return response.choices[0].message.content.strip()

    def get_token_usage(self):
        return self.token_usage.copy()
