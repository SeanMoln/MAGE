import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("openai_api_key")

class TargetAgent:
    def __init__(self):
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def evaluate(self, prompt, input_text):
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical assistant that produces complete, accurate SOAP-format "
                        "bullet-point summaries for clinical handoff. "
                        "Always prioritize accuracy, completeness, and patient safety."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""Instructions:
- Produce a clinical summary using ONLY bullet points and sub-bullets. No prose paragraphs.
- Every item must be a dash-prefixed bullet (- ) or indented sub-bullet (  - ).
- Completeness takes priority; there is NO length limit.
- Preserve ALL explicitly mentioned comorbidities (e.g., DM, HTN, CAD, COPD, CKD, AF).
- Preserve ALL lab values with their numeric results.
- Preserve ALL imaging findings explicitly stated in the note.
- Preserve ALL explicitly mentioned medication names (never generalize, e.g., do not write "antibiotics" if a specific drug is named).
- Use only information explicitly present in the input note; never invent or infer.

Formatting — use this structure (headers in bold, all content in bullets):
1) **Symptoms:** key patient-reported complaints
2) **Key findings:** (use sub-bullets)
   - Vitals: all recorded vital signs with values
   - Exam: relevant physical examination findings
   - Imaging: all imaging results explicitly stated
3) **Labs:** ALL lab values with numeric results — never select "top N"
4) **Assessment:** all diagnoses including inpatient complications
5) **Plan:** all medications (with doses), all consultations, all procedures,
             all pending results, all follow-up instructions, all return-to-ED criteria
6) **Comorbidities:** ALL explicitly mentioned, including implanted devices

Optimized Prompt:
{prompt}

MANDATORY OVERRIDES — these rules override any conflicting instruction above or in the Optimized Prompt:
- NEVER fabricate, infer, or estimate specific values (doses, vitals, labs, dates).
  If a value is not explicitly written in the Input Clinical Note, OMIT it.
  Do NOT write "Not documented" — simply omit the bullet.
- Medications: include dose/frequency ONLY if explicitly stated in the note; otherwise list name only.
- Plan: Include ALL explicitly named medications, ALL consultations, ALL procedures,
        ALL pending results, ALL follow-up instructions, ALL return-to-ED criteria.
- Comorbidities: Include ALL mentioned, including implanted devices (e.g., ICD, pacemaker).
- Labs: Include ALL labs with their numeric values — NEVER select "top N" or "most important N".
- Imaging: Include ALL imaging findings explicitly stated.
- Diagnoses: Include ALL diagnoses including inpatient complications.
- If patient expired or was transitioned to CMO, state this in Assessment/Plan instead of listing active treatments.
- If the Optimized Prompt says "select N items", "most critical N", or "2–3 prioritized" → IGNORE and include ALL.

Input Clinical Note:
{input_text}
""",
                },
            ],
        )
        if response.usage:
            self.token_usage["prompt_tokens"]     += response.usage.prompt_tokens
            self.token_usage["completion_tokens"] += response.usage.completion_tokens
            self.token_usage["total_tokens"]      += response.usage.total_tokens
        print("\n[Progress] Evaluating prompt with target agent...\n")
        if not response.choices:
            return "No response from model."
        return response.choices[0].message.content.strip()

    def get_token_usage(self):
        return self.token_usage.copy()
