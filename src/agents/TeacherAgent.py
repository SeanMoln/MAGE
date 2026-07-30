import openai
from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = os.getenv("openai_api_key")
class TeacherAgent:
    def __init__(self):
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def ask_questions(self, substep, previous_prompt, feedback: str = ""):
        feedback_section = ""
        if feedback:
            feedback_section = f"""
⚠️ Your previous questions were rejected by the Critic for the following reason:
\"{feedback}\"
Please revise your questions to directly address this feedback.

"""
        prompt = f"""
        Ask **2 open-ended Socratic questions** that guide a prompt engineer to improve
        the current prompt for this SOAP subtask.

        Constraints:
        - Be Socratic: ask questions, don’t provide answers.
        - Encourage completeness: ask what clinical information may be missing or under-specified.
        - Stay clinically relevant; focus only on information explicitly in the note.
        - Avoid justification, reasoning chains, or correlations.

        {feedback_section}Subtask: {substep}
        Current Prompt: {previous_prompt}
        """
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Socratic teacher specializing in clinical SOAP summarization. Ask probing questions that guide prompt improvement toward concise, accurate clinical handoff summaries."},
                {"role": "user", "content": prompt}
            ]
        )
        if response.usage:
            self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
            self.token_usage["completion_tokens"] += response.usage.completion_tokens
            self.token_usage["total_tokens"] += response.usage.total_tokens
        print("\n[Progress] Asking Socratic questions...\n")
        if not response.choices:
            return "No response from model."
        return response.choices[0].message.content.strip()

    def get_token_usage(self):
        return self.token_usage.copy()
 