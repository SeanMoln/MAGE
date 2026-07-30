import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("openai_api_key")


class TargetAgentGap:
    """
    Gap-filling 版 TargetAgent。
    輸出限縮在原文詞數 × 0.65 以下（壓縮比 ≤ 0.65）。
    格式強制條例式（bullet + sub-bullet），不允許散文段落。
    保留 Baseline 所有資訊，並依 Student gap prompt 補充缺漏。
    """

    def __init__(self):
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def evaluate(self, prompt: str, input_text: str, baseline_draft: str = "") -> str:
        """
        Parameters
        ----------
        prompt         : StudentAgentGap 精煉後的缺漏補充指令
        input_text     : 原始病歷（事實來源）
        baseline_draft : Baseline TargetAgent 的初稿（參照，確保已有內容不遺失）
        """
        baseline_words = len(baseline_draft.split()) if baseline_draft else 0
        max_words = max(baseline_words + 50, 250, int(len(input_text.split()) * 0.65))

        baseline_section = ""
        if baseline_draft:
            baseline_section = f"""
Baseline Summary (reference — all information here must appear in the final output):
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
                        "You are a clinical assistant that produces complete, accurate SOAP-format "
                        "bullet-point summaries for clinical handoff. "
                        "Always prioritize accuracy, completeness, and patient safety."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""Your task is to produce a complete clinical summary that incorporates the baseline content and fills in any gaps identified by the gap-filling instructions.

{baseline_section}Gap-Filling Instructions (additional information to include beyond the baseline):
{prompt}

═══════════════════════════════════════════════════════
ABSOLUTE RULE — PATIENT SAFETY:
Never fabricate, infer, or estimate any specific value.
This includes: medication doses, frequencies, vital signs,
lab values, imaging measurements, dates, or quantities.
If a specific value is NOT explicitly written in the
Original Clinical Note below, OMIT it entirely.
Do NOT write "Not documented", "unknown", or placeholder text.
Silence is safer than a fabricated number.
═══════════════════════════════════════════════════════

MANDATORY FORMAT RULES:
- Use ONLY bullet points and sub-bullets. No prose paragraphs.
- Every item must be a dash-prefixed bullet (- ) or indented sub-bullet (  - ).
- NEVER write a section header with no content following it.
- Structure (bold headers, all content in bullets):
  1) **Symptoms:** key patient-reported complaints
  2) **Key findings:** Vitals / Exam / Imaging
     — ONLY include vital sign values that appear verbatim in the note
  3) **Labs:** ONLY lab values with numeric results stated in the note
  4) **Assessment:**
     — If the patient expired or was transitioned to comfort measures (CMO),
       state this explicitly and do NOT list ongoing active treatment plans
     — Otherwise list all diagnoses including inpatient complications
  5) **Plan:**
     — Medications: name + dose/frequency ONLY if explicitly stated in the note;
       if dose is absent, write the medication name only (e.g., "- Amiodarone")
     — All consultations ordered
     — All procedures performed or planned
     — Follow-up: include specifics only if stated (date/location/specialty)
     — Return-to-ED criteria if mentioned
     — Pending results if mentioned
  6) **Comorbidities:** ALL explicitly mentioned, including implanted devices
- Preserve ALL information from the Baseline Summary above.
- Add information from Gap-Filling Instructions not already present.
- WORD LIMIT: The entire output MUST NOT exceed {max_words} words.

Original Clinical Note:
{input_text}
""",
                },
            ],
        )
        if response.usage:
            self.token_usage["prompt_tokens"]     += response.usage.prompt_tokens
            self.token_usage["completion_tokens"] += response.usage.completion_tokens
            self.token_usage["total_tokens"]      += response.usage.total_tokens
        print("\n[Gap Target] 依 Student prompt 生成完整摘要...\n")
        if not response.choices:
            return "No response from model."
        return response.choices[0].message.content.strip()

    def get_token_usage(self):
        return self.token_usage.copy()
