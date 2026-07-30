# evaluate_summary.py
# -----------------------------------------------------------
# Evaluate summaries from results.json (JSON array or JSONL).
# Implements hybrid weighted scoring per paper formula:
#   H = λ * R_rule + (1 - λ) * R_semantic
# Facets (S/O/A/P) and Key Elements (dx, labs, imaging, comorbidities, plan)
# -----------------------------------------------------------

import re
import os
import json
import argparse
import csv
import openai
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional,Tuple 

# =========================
# Heuristic dictionaries
# =========================
LAB_PATTERNS = [
    r"\bWBC\b", r"\bHb\b", r"\bHgb\b", r"\bNa\b", r"\bK\b",
    r"\bCr\b|\bcreatinine\b", r"\bBNP\b", r"\btroponin\b",
    r"\bAST\b", r"\bALT\b", r"\bbilirubin\b|\bT[- ]?bil\b",
    r"\bCRP\b", r"\bhCG\b", r"\bHbA1c\b", r"\bABG\b",
    r"\bpH\b", r"\bpCO2\b", r"\bpO2\b", r"\bbicarb\b|\bHCO3\b"
]

IMAGING_KEYWORDS = [
    "x-ray", "xray", "chest x-ray", "cxr",
    "ct", "cta", "ct scan", "ct chest", "ct abdomen", "ct head", "head ct", "ctpa",
    "mri", "mra", "mri brain", "mri spine",
    "echocardiogram", "echo", "tte", "tee",
    "angiography",
    "ultrasound", "ultrasonography",
    "doppler",
    "radiograph", "radiography",
    "sonogram", "sonography"
]

COMORBIDITY_KEYWORDS = [
    "diabetes","dm","hypertension","htn","chronic kidney disease","ckd",
    "chronic renal insufficiency","chronic renal failure","renal insufficiency",
    "heart failure","hf","coronary artery disease","cad","copd","asthma",
    "atrial fibrillation","af ","stroke","tia","cancer","malignancy",
    "immunosuppression","liver disease","cirrhosis","obesity",
    "hyperlipidemia","dyslipidemia","hypercholesterolemia","ckd stage",
    "gout","anemia","osteoarthritis","arthritis","gerd","reflux",
    "osteoporosis","hypothyroidism","hyperthyroidism","peripheral vascular disease",
    "pvd","depression","anxiety","epilepsy","seizure",
]

PLAN_KEYWORDS_GENERAL = [
    "admit","admission","oxygen","o2","npo",
    "iv fluids","analgesic","analgesics","pain control",
    "antibiotic","antibiotics","ceftriaxone","metronidazole","azithromycin","nitrofurantoin",
    "diuretic","furosemide","insulin",
    "consult","surgical consult","cardiology consult","ortho consult","nephrology consult","obgyn consult",
    "pci","nitroglycerin","rate control","diltiazem","beta blocker","anticoagulation","heparin",
    "tpa","thrombolysis","follow-up","follow up","fu","monitor","telemetry",
    "referral","start","initiate","begin","titrate","hold","discontinue",
    "sertraline","cefazolin","vancomycin","piperacillin-tazobactam","amoxicillin","prescribe","administer","give","continue","switch"
]
PLAN_MINIMUMS_ADHF = ["fluid restriction","daily weights"]

DIAGNOSIS_HINT_WORDS = [
    "diagnosis","dx","impression","assessment",
    "stemi","nstemi","myocardial infarction","heart failure","acute decompensated heart failure",
    "atrial fibrillation","af with rvr","pneumonia","sepsis",
    "acute cholecystitis","appendicitis","peptic ulcer","acute hepatitis",
    "copd exacerbation","uti","urinary tract infection","dka","diabetic ketoacidosis",
    "aki","acute kidney injury","ischemic stroke","stroke","gad","generalized anxiety disorder"
]

SYMPTOM_WORDS = [
    "pain","shortness of breath","dyspnea","nausea","vomit","fever","cough",
    "palpitation","palpitations","fatigue","orthopnea","edema","chest","dysuria","weakness","lightheadedness",
    "sputum","productive cough","productive sputum","anorexia","appetite loss","wheezes","hematuria","polyuria","polydipsia"
]

# =========================
# Utilities
# =========================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _word_count(s: str) -> int:
    return len(re.findall(r"\w+", s))

def _contains_any(text: str, keywords: List[str]) -> List[str]:
    t = _norm(text)
    hits = []
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", t):
            hits.append(kw)
    return sorted(list(set(hits)))

def _contains_any_regex(text: str, patterns: List[str]) -> List[str]:
    hits = []
    for pat in patterns:
        if re.search(pat, text, flags=re.I):
            hits.append(pat)
    return sorted(list(set(hits)))

def split_sentences(text: str) -> List[str]:
    # 只在句號後有空格、換行，或分號後分割，避免過度分割
    parts = re.split(r'\.\s+|;\s*\n|[\n]{2,}', text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]

def bullets_from_summary(summary_text: str) -> List[str]:
    lines = [l.strip(" -\t") for l in summary_text.splitlines() if l.strip()]
    if len(lines) <= 1:
        return split_sentences(summary_text)
    return [l for l in lines if l]

# =========================
# SOAP parsing
# =========================
def _extract_section(text: str, names: List[str]) -> Optional[str]:
    """
    Extract the first section that matches any of `names`.
    Works for:
      - Header alone on its own line
      - Header followed by inline content on the same line, e.g. 'S: chest pain...'
    Stops when the next SOAP-like header is encountered.
    """
    # normalize target names (lowercase, trim ':')
    nm = [n.lower().rstrip(":") for n in names]
    # accepted aliases (EN only)
    aliases = []
    for n in nm:
        if n in ("s", "subjective"):
            aliases += ["s", "subjective"]
        elif n in ("o", "objective"):
            aliases += ["o", "objective"]
        elif n in ("a", "assessment", "impression"):
            aliases += ["a", "assessment", "impression"]
        elif n in ("p", "plan"):
            aliases += ["p", "plan"]
        else:
            aliases.append(n)
    aliases = sorted(set(a.lower().rstrip(":") for a in aliases), key=len, reverse=True)

    # header pattern (EN only), allow optional ':' or '-' and optional inline content
    header = re.compile(
        rf'^\s*((?:{"|".join(map(re.escape, aliases))}))(?=\s*[:\-]|\s*$)\s*[:\-]?\s*(.*)\s*$',
        re.I,
    )
    # any SOAP header (for stopping)
    any_header = re.compile(r'^\s*(?:s|o|a|p|subjective|objective|assessment|impression|plan)\s*[:\-]?\s*(.*)\s*$', re.I)

    lines = text.splitlines()
    buf = []
    capturing = False

    for raw in lines:
        m = header.match(raw)
        if m:
            # starting the target section
            if capturing and buf:
                break
            capturing = True
            inline = m.group(2).strip()
            if inline:
                buf.append(inline)
            continue

        if capturing:
            # stop when we see another SOAP-ish header line
            if any_header.match(raw):
                break
            buf.append(raw)

    out = "\n".join(buf).strip()
    return out if out else None


SOAP_ALIAS_MAP = {
    "s": "S",
    "subjective": "S",
    "cc": "S",
    "chief complaint": "S",
    "hpi": "S",
    "history of present illness": "S",
    "history": "S",
    "o": "O",
    "objective": "O",
    "physical exam": "O",
    "pe": "O",
    "exam": "O",
    "vital signs": "O",
    "vitals": "O",
    "labs": "O",
    "laboratory": "O",
    "imaging": "O",
    "a": "A",
    "assessment": "A",
    "impression": "A",
    "diagnosis": "A",
    "diagnoses": "A",
    "dx": "A",
    "problem list": "A",
    "p": "P",
    "plan": "P",
    "disposition": "P",
    "discharge": "P",
    "recommendations": "P",
    "follow-up": "P",
    "follow up": "P",
}

SECTION_ALIASES = {
    "S": ["subjective", "s", "chief complaint", "cc", "history of present illness", "hpi", "history"],
    "O": ["objective", "o", "physical exam", "pe", "exam", "vital signs", "vitals", "labs", "laboratory", "imaging"],
    "A": ["assessment", "a", "impression", "diagnosis", "diagnoses", "dx", "problem list"],
    "P": ["plan", "p", "disposition", "discharge", "recommendations", "follow-up", "follow up"],
}


def _clean_section_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip(" :-\n\t")
    return text


def _extract_marked_sections(text: str) -> Dict[str, str]:
    """
    Parse inline/block section markers such as:
      - S: ... O: ... A: ... P: ...
      - HPI: ...
      - Physical Exam: ...
      - Assessment: ...
    """
    aliases = sorted(SOAP_ALIAS_MAP, key=len, reverse=True)
    marker_re = re.compile(
        rf"(?<!\w)(?P<label>{'|'.join(map(re.escape, aliases))})\s*:\s*",
        flags=re.I,
    )
    sections = {"S": [], "O": [], "A": [], "P": []}
    matches = list(marker_re.finditer(text))
    for idx, match in enumerate(matches):
        label = SOAP_ALIAS_MAP[match.group("label").lower()]
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = _clean_section_text(text[start:end])
        if chunk:
            sections[label].append(chunk)
    return {key: _clean_section_text(" ".join(value)) for key, value in sections.items()}


def _extract_heading_sections(text: str) -> Dict[str, str]:
    sections = {}
    for key, names in SECTION_ALIASES.items():
        chunk = _extract_section(text, names)
        sections[key] = _clean_section_text(chunk or "")
    return sections


def _fallback_section_split(text: str) -> Dict[str, str]:
    """
    Fallback for loosely structured notes:
      - before assessment/impression -> S/O
      - assessment/impression block -> A
      - plan/disposition/follow-up block -> P
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    out = {"S": "", "O": "", "A": "", "P": ""}

    def usable_pre_section(pre: str) -> bool:
        if not pre:
            return False
        return re.match(r"^(?:s|o|a|p|subjective|objective|assessment|impression|plan)\s*:", pre, flags=re.I) is None

    assess_match = re.search(
        r"\b(assessment|impression|diagnosis|diagnoses|dx)\s*:\s*",
        normalized,
        flags=re.I,
    )
    plan_match = re.search(
        r"\b(plan|disposition|follow[- ]?up|recommendations?)\s*:\s*",
        normalized,
        flags=re.I,
    )

    if assess_match:
        pre = normalized[:assess_match.start()].strip()
        if usable_pre_section(pre):
            out["S"] = pre
            out["O"] = pre
        if plan_match and plan_match.start() > assess_match.end():
            out["A"] = _clean_section_text(normalized[assess_match.end():plan_match.start()])
            out["P"] = _clean_section_text(normalized[plan_match.end():])
        else:
            out["A"] = _clean_section_text(normalized[assess_match.end():])
    elif plan_match:
        pre = normalized[:plan_match.start()].strip()
        if usable_pre_section(pre):
            out["S"] = pre
            out["O"] = pre
        out["P"] = _clean_section_text(normalized[plan_match.end():])
    else:
        out["S"] = normalized
        out["O"] = normalized

    return {key: _clean_section_text(value) for key, value in out.items()}


def parse_soap_sections(soap_text: str) -> Dict[str, str]:
    """
    Parse SOAP-like sections from free-text clinical notes.
    Priority:
      1. Explicit inline/block section labels (S:/O:/A:/P:, HPI:, Exam:, ...)
      2. Standalone headings on separate lines
      3. Fallback split using assessment/plan anchors
    """
    text = (soap_text or "").strip()
    if not text:
        return {"S": "", "O": "", "A": "", "P": ""}

    best_sections = {"S": "", "O": "", "A": "", "P": ""}
    best_count = 0
    for parser in (_extract_marked_sections, _extract_heading_sections, _fallback_section_split):
        sections = parser(text)
        section_count = sum(bool(v) for v in sections.values())
        if section_count > best_count:
            best_sections = sections
            best_count = section_count
        if section_count >= 2:
            return sections

    if best_count > 0:
        return best_sections

    return {"S": _clean_section_text(text), "O": _clean_section_text(text), "A": "", "P": ""}







DIAG_LINE_PAT = re.compile(r"^\s*(impression|assessment|diagnosis|dx)\s*[:\-]\s*(.+)$", flags=re.I)

def detect_main_diagnoses_from_assessment(assessment_text: str) -> List[str]:
    a = assessment_text.strip()
    hits = set()
    for line in a.splitlines():
        m = DIAG_LINE_PAT.match(line.strip())
        if m:
            payload = m.group(2).strip().rstrip(".;")
            for chunk in re.split(r"[;,/]| and ", payload, flags=re.I):
                chunk = chunk.strip().lower()
                if chunk:
                    hits.add(chunk)
    a_norm = _norm(a)
    for word in DIAGNOSIS_HINT_WORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", a_norm):
            hits.add(word.lower())
    if not hits and a_norm:
        m = re.match(r"([a-z0-9 \-/&()]+?)([.;]|$)", a_norm)
        if m:
            first = m.group(1).strip()
            if first:
                hits.add(first)
    return sorted(list(set([h for h in hits if len(h) >= 3])))

def extract_key_items_from_soap(soap: Dict[str, str]) -> Dict[str, List[str]]:
    S = (soap.get("S") or "").lower()
    O = (soap.get("O") or "").lower()
    A = (soap.get("A") or "").lower()
    P = (soap.get("P") or "").lower()

    # Labs (EN)
    LAB_PATTERNS = [
        r"\bwbc\b", r"\bhb\b|\bhgb\b", r"\bna\b", r"\bk\b",
        r"\bcr\b|\bcreatinine\b", r"\bbnp\b", r"\btroponin\b",
        r"\bast\b|\bsgot\b", r"\balt\b|\bsgpt\b", r"\bbilirubin\b|\bt[- ]?bil\b",
        r"\bcrp\b", r"\bhba1c\b", r"\babg\b", r"\bpH\b",
        r"\blactate\b", r"\belectrolyte(s)?\b", r"\blft(s)?\b", r"\brft(s)?\b",
        r"\bhemoglobin\b", r"\bhematocrit\b",
        r"\bwbc\b|\bwhite blood\b|\bleukocyte(s)?\b",
        r"\bplatelet(s)?\b|\bplt\b",
        r"\bpotassium\b", r"\bsodium\b", r"\bglucose\b", r"\burea\b|\bbun\b"
    ]

    # Imaging (EN) — avoid 'us' false positives; allow echo/CT/MRI/CXR
    IMG_PATTERNS = [
        r"\bx[- ]?ray\b", r"\bxray\b", r"\bcxr\b", r"\bchest x[- ]?ray\b",
        r"(?<![a-z])ct(?![a-z])", r"\bcta\b", r"(?<![a-z])mri(?![a-z])", r"\bmra\b",
        r"\becho(cardio(gram)?)?\b", r"\bultrasound\b", r"\bultrasonograph(y|ic)?\b",
        r"\bangiograph(y|ic)?\b", r"\bradiograph(y|ic|s)?\b",
        r"\bdoppler\b", r"\bsonograph(y|ic)?\b", r"\bsonogram\b",
    ]

    # Diagnoses (A-section hints)
    DX_HINTS = [r"\bdiagnosis(es)?\b", r"\bassessment\b", r"\bimpression\b"]

    # Comorbidities (common EN)
    COMORB_HINTS = [
        r"\bdm\b|\bdiabetes\b", r"\bhtn\b|\bhypertension\b",
        r"\bckd\b|\bchronic kidney\b", r"\bcopd\b", r"\basthma\b",
        r"\baf\b|\batrial fibrillation\b", r"\bcad\b|\bcoronary artery disease\b",
        r"\bhf\b|\bheart failure\b", r"\badhf\b"
    ]

    # Plan (EN) — 擴充 follow-up 相關模式
    PLAN_HINTS = [
        r"\bplan\b", r"\bdiuretic(s)?\b", r"\bantibiotic(s)?\b", r"\bfluid(s)?\b",
        r"\brestrict\b", r"\bfollow(?:ed|ing)?[- ]?up\b", r"\border\b", r"\bconsult(?:ation|ed|ing)?\b",
        r"\badmiss?ion\b|\badmit(?:ted|ting)?\b|\bdischarge[d]?\b|\bobservation\b",
        # 擴充：接受 PCP、outpatient、clinic、appointment 等也算有 follow-up 規劃
        r"\bpcp\b", r"\boutpatient\b", r"\bclinic\b", r"\bappointment\b",
        r"\breturn[- ]?to[- ]?ed\b|\bcome[- ]?back\b|\bif[- ]?worse\b"
    ]

    def _find(patterns, text):
        hits = []
        for pat in patterns:
            if re.search(pat, text, flags=re.I):
                hits.append(pat)
        return hits

    dx   = _find(DX_HINTS, A) or _find(DX_HINTS, S)
    labs = _find(LAB_PATTERNS, O)
    img  = _find(IMG_PATTERNS, O + " " + A + " " + P + " " + S)
    # 共病只從 A（出院診斷/Assessment）和 S（主訴/PMH）段偵測
    # 避免把 O 段（住院病程）的術後併發症誤判為既有共病
    com  = _find(COMORB_HINTS, A + " " + S)
    plan = _find(PLAN_HINTS, P)

    return {
        "diagnoses": dx,
        "labs": labs,
        "imaging": img,
        "comorbidities": com,
        "plan_general": plan,
        "plan_minimums_adhf": _find([r"\badhf\b|\bacute decompensated heart failure\b"], A + " " + P),
    }


# =========================
# Embeddings (semantic)
# =========================
def get_embeddings(texts: List[str]):
    try:
        import numpy as np
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()

        # 讀取 API key 並傳遞給 OpenAI 客戶端
        api_key = os.getenv("openai_api_key")
        if not api_key:
            print("[get_embeddings] WARNING: openai_api_key not found in environment")
            return None

        client = OpenAI(api_key=api_key)  # 明確傳遞 API key
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        vecs = [d.embedding for d in resp.data]
        return np.array(vecs, dtype="float32")
    except Exception as e:
        # 臨時加上除錯輸出，幫你定位問題
        print("[get_embeddings] ERROR:", repr(e))
        return None


def cosine_sim(a, b):
    import numpy as np
    a = a.astype("float32"); b = b.astype("float32")
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T

def extract_candidates_semantic(original_text: str) -> Dict[str, List[str]]:
    sents = split_sentences(original_text)
    S_cand, O_cand, A_cand, P_cand = [], [], [], []
    for s in sents:
        s_low = s.lower()
        if any(w in s_low for w in SYMPTOM_WORDS):
            S_cand.append(s)
        if any(re.search(p, s, flags=re.I) for p in [r"\btemp", r"\bbp\b|\bblood pressure\b",
                                                     r"\bhr\b|\bheart rate\b", r"\brr\b|\brespiratory rate\b",
                                                     r"\bspo2\b|\bo2\b|\boxygen\b"]) \
           or _contains_any_regex(s, LAB_PATTERNS) or bool(_contains_any(s, IMAGING_KEYWORDS)):
            O_cand.append(s)
        if re.search(r'\b(impression|assessment|diagnosis|dx)\b', s, flags=re.I) \
           or any(k in s_low for k in DIAGNOSIS_HINT_WORDS):
            A_cand.append(s)
        if _contains_any(s, PLAN_KEYWORDS_GENERAL + PLAN_MINIMUMS_ADHF):
            P_cand.append(s)
    return {
        "S": sorted(set(S_cand)),
        "O": sorted(set(O_cand)),
        "A": sorted(set(A_cand)),
        "P": sorted(set(P_cand)),
    }

# =============== Rule Scores ===============
def soap_retention_rule(soap: Dict[str, str], summary_text: str) -> Dict[str, float]:
    s_sum = (summary_text or "").lower()

    # 源文是否「存在」S/O/A/P（由 Auto-SOAP parser 分段後是否非空）
    S_exist = bool((soap.get("S") or "").strip())
    O_exist = bool((soap.get("O") or "").strip())
    A_exist = bool((soap.get("A") or "").strip())
    P_exist = bool((soap.get("P") or "").strip())

    # 摘要中的字面線索（加入你的格式）
    hints_S = [
        r"\bsubjective\b", r"\bs:\b",
        r"\bsymptom(s)?\b",                      # 新增
        r"^\s*[-*]\s*\*\*symptoms?\*\*",        # 新增：你的粗體項目
    ]
    hints_O = [
        r"\bobjective\b", r"\bo:\b",
        r"\bkey findings?\b",                    # 新增：你的標題
        r"\bvital(s)?\b", r"\blab(s)?\b",
        r"\becg\b", r"\bcxr\b", r"\bx[- ]?ray\b",
        r"(?<![a-z])ct(?![a-z])", r"(?<![a-z])mri(?![a-z])", r"\bultrasound\b"
    ]
    hints_A = [r"\bassessment\b", r"\ba:\b", r"\bdiagnosis(es)?\b", r"\bimpression\b"]
    hints_P = [r"\bplan\b", r"\bp:\b", r"\bfollow[- ]?up\b", r"\bconsult\b", r"\border\b", r"\btreat(ment)?\b"]

    def any_hint(hints):
        return any(re.search(h, s_sum, flags=re.I|re.M) for h in hints)

    # 只在源文該段落「確實存在」時才判定保留（避免 NA 被當 0 分）
    return {
        "S": 1.0 if S_exist and any_hint(hints_S) else (1.0 if not S_exist else 0.0),
        "O": 1.0 if O_exist and any_hint(hints_O) else (1.0 if not O_exist else 0.0),
        "A": 1.0 if A_exist and any_hint(hints_A) else (1.0 if not A_exist else 0.0),
        "P": 1.0 if P_exist and any_hint(hints_P) else (1.0 if not P_exist else 0.0),
    }

def key_elements_rule(soap: Dict[str, str], summary_text: str) -> Tuple[Dict[str, Dict[str, bool]], Dict[str, float]]:
    key_items = extract_key_items_from_soap(soap)
    presence = {k:{} for k in key_items}
    s_norm = _norm(summary_text)
    # 所有 key_items 的值都是 regex pattern（來自 COMORB_HINTS / DX_HINTS 等）
    # 直接用 re.search(v, ...) 而不做 re.escape，避免把 pattern 當字面字串
    for k, m in key_items.items():
        for v in m:
            ok = re.search(v, summary_text, flags=re.I) is not None
            presence[k][v] = bool(ok)
    recall = {}
    for k, m in key_items.items():
        if not m:
            recall[k] = 1.0
        else:
            got = sum(1 for v in m if presence[k].get(v, False))
            recall[k] = round(got / len(m), 4)
    return presence, recall

# =============== Semantic Scores ===============
def facet_semantic_score(original_text: str, bullets: List[str], tau: float, embeddings_ok: bool) -> Dict[str, float]:
    if not embeddings_ok:
        # No embeddings available -> semantic = 0 (paper formula will rely on rule)
        return {"S":0.0, "O":0.0, "A":0.0, "P":0.0}
    cands = extract_candidates_semantic(original_text)
    # ✅ 優化：只計算一次 bullets embeddings
    B = get_embeddings(bullets) if bullets else None
    if B is None:
        return {"S":0.0, "O":0.0, "A":0.0, "P":0.0}

    scores = {}
    for facet in ["S","O","A","P"]:
        if not cands[facet]:
            scores[facet] = 1.0
            continue
        F = get_embeddings(cands[facet])
        if F is None:
            scores[facet] = 0.0
            continue
        sims = cosine_sim(F, B)  # [Nf, Nb]
        # coverage: proportion of facet sentences that have max similarity >= tau
        covered = (sims.max(axis=1) >= tau).mean()
        scores[facet] = float(covered)
    return scores

def elements_semantic_score(original_text: str, bullets: List[str], tau: float, embeddings_ok: bool) -> Dict[str, float]:
    if not embeddings_ok:
        return {"diagnoses":0.0,"labs":0.0,"imaging":0.0,"comorbidities":0.0,"plan_general":0.0,"plan_minimums_adhf":0.0}
    cands = extract_candidates_semantic(original_text)
    elem2sents = {
        "diagnoses": cands["A"],
        "labs": [s for s in cands["O"] if _contains_any_regex(s, LAB_PATTERNS)],
        "imaging": [s for s in (cands["O"] + cands["S"] + cands["A"] + cands["P"]) if _contains_any(s, IMAGING_KEYWORDS)],
        # 共病候選句：從全文直接掃（PMH、Assessment 都能命中），不侷限於 S/A 分類結果
        "comorbidities": [s for s in split_sentences(original_text) if _contains_any(s, COMORBIDITY_KEYWORDS)],
        "plan_general": cands["P"],
        "plan_minimums_adhf": [s for s in cands["P"] if _contains_any(s, PLAN_MINIMUMS_ADHF)],
    }
    # ✅ 優化：只計算一次 bullets embeddings
    B = get_embeddings(bullets) if bullets else None
    if B is None:
        return {"diagnoses":0.0,"labs":0.0,"imaging":0.0,"comorbidities":0.0,"plan_general":0.0,"plan_minimums_adhf":0.0}

    out = {}
    for k, sents in elem2sents.items():
        if not sents:
            out[k] = 1.0
            continue
        F = get_embeddings(sents)
        if F is None:
            out[k] = 0.0
            continue
        sims = cosine_sim(F, B)
        out[k] = float((sims.max(axis=1) >= tau).mean())
    return out

# =============== Hybrid Integration ===============
def hybrid_score(rule_val: float, sem_val: float, lam: float) -> float:
    return float(max(rule_val, sem_val))

@dataclass
class EvalReport:
    compression_ratio: float
    compression_pass: bool
    critical_miss: bool
    soap_retention: Dict[str, float]
    key_element_recall: Dict[str, float]
    details: Dict[str, Dict]

def evaluate_pair(
    original_soap_text: str,
    bullet_summary_text: str,
    mode: str = "hybrid",
    lambda_facet: float = 0.6,
    lambda_element: float = 0.6,
    theta_facet: float = 0.6,
    theta_element: float = 0.6,
    tau_facet: float = 0.60,
    tau_element: float = 0.62,
    comp_limit: float = 0.60,   # ← 調整預設門檻為 0.6
    w_orig_override: int = 0,   # ← 若 > 0，用此值取代從原文計算的字數（供截斷文本使用）
) -> EvalReport:

    # ---- 1) 字數與壓縮比 ----------------------------------------------------
    w_orig = w_orig_override if w_orig_override > 0 else _word_count(original_soap_text)
    w_sum  = _word_count(bullet_summary_text)
    compression_ratio = round((w_sum / w_orig), 4) if w_orig else 0.0
    compression_pass  = (compression_ratio <= comp_limit) if w_orig else True

    # ---- 2) 解析原文 SOAP 與摘要 bullets ------------------------------------
    soap    = parse_soap_sections(original_soap_text)

    bullets = bullets_from_summary(bullet_summary_text)

    # ---- 3) 檢查 embeddings 可用性（僅 semantic/hybrid 需要）---------------
    embeddings_ok = False
    if mode in ("semantic", "hybrid"):
        try:
            embeddings_ok = (get_embeddings(["ping"]) is not None)
        except Exception:
            embeddings_ok = False

    # Hybrid 但沒有 embeddings → 自動回退為 rule，並避免 0.6 常數效應
    if mode == "hybrid" and not embeddings_ok:
        mode = "rule"
        lambda_facet = 1.0
        lambda_element = 1.0

    # ---- 4) Facet（S/O/A/P）計分：規則、語義、混合 --------------------------
    rule_f = soap_retention_rule(soap, bullet_summary_text)
    sem_f  = facet_semantic_score(
        original_soap_text, bullets, tau=tau_facet, embeddings_ok=embeddings_ok
    )

    # ---- 5) 要素召回：規則、語義、混合（含 NA 規則）-------------------------
    # key_elements_rule() 需回傳：(presence_rule, recall_rule)
    # - presence_rule[k]：原文是否存在該要素（True/False）
    # - recall_rule[k]  ：規則召回分數（0~1）
    presence_rule, recall_rule = key_elements_rule(soap, bullet_summary_text)
    recall_sem = elements_semantic_score(
        original_soap_text, bullets, tau=tau_element, embeddings_ok=embeddings_ok
    )

    elem_keys = ["diagnoses", "labs", "imaging", "comorbidities", "plan_general", "plan_minimums_adhf"]

    # ---- 6) 模式整合 --------------------------------------------------------
    if mode == "rule":
        facet_scores = rule_f
        # NA → 1.0；其餘用規則值
        elem_scores = {
            k: (1.0 if not presence_rule.get(k, False) else float(recall_rule.get(k, 0.0)))
            for k in elem_keys
        }

    elif mode == "semantic":
        facet_scores = sem_f
        # NA → 1.0；其餘用語義值
        elem_scores = {
            k: (1.0 if not presence_rule.get(k, False) else float(recall_sem.get(k, 0.0)))
            for k in elem_keys
        }

    else:
        # hybrid：NA 直接 1.0（不做加權）；非 NA 用 λ 加權
        facet_scores = {
            k: hybrid_score(rule_f.get(k, 0.0), sem_f.get(k, 0.0), lambda_facet) for k in ["S", "O", "A", "P"]
        }
        elem_scores = {}
        for k in elem_keys:
            if not presence_rule.get(k, False):
                elem_scores[k] = 1.0
            else:
                r = float(recall_rule.get(k, 0.0))
                s = float(recall_sem.get(k, 0.0))
                elem_scores[k] = hybrid_score(r, s, lambda_element)

    # ---- 7) SOAP 保留率：連續語義分數（方案 A）--------------------------------
    # 以語義覆蓋率作為連續值（0.0~1.0），不做門檻二值化；
    # 原文該段落不存在（NA）時 sem_f 已給 1.0，embeddings 不可用時回退 rule_f。
    facet_pass = sem_f if embeddings_ok else rule_f

    # critical miss：若 diagnoses 或 plan_general 低於 θ_element
    critical_miss = False
    if elem_scores.get("diagnoses", 1.0) < theta_element or elem_scores.get("plan_general", 1.0) < theta_element:
        critical_miss = True

    # ---- 8) 結果包裝 --------------------------------------------------------
    details = {
        "word_counts": {"original": w_orig, "summary": w_sum},
        "facet_rule": rule_f,
        "facet_semantic": sem_f,
        "element_rule_presence": presence_rule,
        "element_rule_recall": recall_rule,
        "element_semantic_recall": recall_sem,
        "params": {
            "mode": mode,
            "lambda_facet": lambda_facet,
            "lambda_element": lambda_element,
            "theta_facet": theta_facet,
            "theta_element": theta_element,
            "tau_facet": tau_facet,
            "tau_element": tau_element,
            "comp_limit": comp_limit,
            "embeddings_ok": embeddings_ok,
        },
    }

    return EvalReport(
        compression_ratio=float(compression_ratio),
        compression_pass=bool(compression_pass),
        critical_miss=bool(critical_miss),
        soap_retention=facet_pass,
        key_element_recall=elem_scores,
        details=details,
    )



# =========================
# CLI
# =========================
def main():
    import argparse, csv

    ap = argparse.ArgumentParser(
        description="Evaluate summaries from results.json (array or JSONL)."
    )
    ap.add_argument("--json", type=str, required=True,
                    help="Path to results.json (array or JSONL).")
    ap.add_argument("--out", type=str, default="",
                    help="Optional JSON report path")
    ap.add_argument("--csv", type=str, default="",
                    help="Optional CSV path for summary table")
    ap.add_argument("--avg", action="store_true",
                    help="Print macro averages across all cases")
    ap.add_argument("--mode", choices=["rule","semantic","hybrid"], default="hybrid",
                    help="Scoring mode")

    # 原本的語義門檻（保留，供 semantic/hybrid 使用）
    ap.add_argument("--tau_facet", type=float, default=0.60,
                    help="Semantic coverage threshold for facets (S/O/A/P)")
    ap.add_argument("--tau_element", type=float, default=0.62,
                    help="Semantic rescue threshold for key elements")

    # === 新增：Hybrid 加權與門檻、壓縮比判準 ===
    ap.add_argument("--lambda_facet", type=float, default=0.6,
                    help="Hybrid weight for facet retention (semantic vs rule)")
    ap.add_argument("--lambda_element", type=float, default=0.6,
                    help="Hybrid weight for key-element recall (semantic vs rule)")
    ap.add_argument("--theta_facet", type=float, default=0.6,
                    help="Hybrid decision threshold for facet scores (0~1)")
    ap.add_argument("--theta_element", type=float, default=0.6,
                    help="Hybrid decision threshold for element recall (0~1)")
    ap.add_argument("--comp_limit", type=float, default=0.65,
                    help="Compression ratio upper bound for pass (<= comp_limit)")
    ap.add_argument("--preprocessed", type=str, default="",
                    help="（選填）preprocessed.json 路徑，用完整原文做 SOAP 解析（比 results.json 的 input_text 更完整）")

    args = ap.parse_args()

    # 載入 preprocessed.json（完整原文，依 case_id 索引）
    preprocessed_full = []
    if args.preprocessed:
        try:
            with open(args.preprocessed, "r", encoding="utf-8") as f:
                preprocessed_full = json.load(f)
        except Exception as e:
            print(f"[警告] 無法載入 preprocessed 檔案：{e}，改用 input_text")

    # 載入 JSON 或 JSONL
    data = []
    with open(args.json, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    reports = []

    # 初始化 CSV（寫標題）
    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "case_id","compression_ratio","compression_pass",
            "S_retained","O_retained","A_retained","P_retained",
            "recall_diagnoses","recall_labs","recall_imaging",
            "recall_comorbidities","recall_plan_general","recall_plan_minimums_adhf",
            "orig_words","sum_words"
        ])

    # 使用 tqdm 顯示進度條
    try:
        from tqdm import tqdm
        data_iter = tqdm(enumerate(data), total=len(data), desc="評估進度")
    except ImportError:
        print("[提示] 安裝 tqdm 可顯示進度條：pip install tqdm")
        data_iter = enumerate(data)

    for i, item in data_iter:
        case_id = item.get("case_id", i)
        # 優先用 preprocessed.json 的完整原文，fallback 用 input_text
        if preprocessed_full and case_id < len(preprocessed_full):
            soap_text = preprocessed_full[case_id]
        else:
            soap_text = item.get("input_text", "")
        sum_text = item.get("final_result", "")
        # 若 results.json 有儲存完整詞數（input_length），優先用它計算壓縮率
        w_orig_override = int(item.get("input_length", 0))
        report = evaluate_pair(
            original_soap_text=soap_text,
            bullet_summary_text=sum_text,
            mode=args.mode,
            tau_facet=args.tau_facet,
            tau_element=args.tau_element,
            lambda_facet=args.lambda_facet,
            lambda_element=args.lambda_element,
            theta_facet=args.theta_facet,
            theta_element=args.theta_element,
            comp_limit=args.comp_limit,
            w_orig_override=w_orig_override,
        )

        rep = asdict(report)
        # 附加衍生欄位：compression_pass
        comp_pass = (rep.get("compression_ratio", 1.0) <= args.comp_limit)
        rep["compression_pass"] = bool(comp_pass)
        rep["case_id"] = i
        reports.append(rep)

        # 逐筆寫入 CSV
        if csv_writer:
            csv_writer.writerow([
                rep["case_id"],
                rep.get("compression_ratio", 0.0),
                int(rep.get("compression_pass", False)),
                rep["soap_retention"].get("S",0.0),
                rep["soap_retention"].get("O",0.0),
                rep["soap_retention"].get("A",0.0),
                rep["soap_retention"].get("P",0.0),
                rep["key_element_recall"].get("diagnoses",1.0),
                rep["key_element_recall"].get("labs",1.0),
                rep["key_element_recall"].get("imaging",1.0),
                rep["key_element_recall"].get("comorbidities",1.0),
                rep["key_element_recall"].get("plan_general",1.0),
                rep["key_element_recall"].get("plan_minimums_adhf",1.0),
                rep["details"]["word_counts"]["original"],
                rep["details"]["word_counts"]["summary"],
            ])
            csv_file.flush()  # 立即寫入磁碟

        # 逐筆更新 JSON（覆蓋寫入）
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(reports, f, indent=2, ensure_ascii=False)

    # 關閉 CSV 檔案
    if csv_file:
        csv_file.close()

    # 螢幕輸出摘要（不輸出完整 JSON，避免太長）
    print(f"\n✅ 評估完成！共 {len(reports)} 筆案例")
    print(f"   CSV: {args.csv if args.csv else '(未指定)'}")
    print(f"   JSON: {args.out if args.out else '(未指定)'}")

    # 選擇性：印出宏平均
    if args.avg and reports:
        n = len(reports)
        def avg(fn): return round(sum(fn(r) for r in reports)/n, 4)
        avg_compress = avg(lambda r: r.get("compression_ratio", 0.0))
        avg_comp_pass = avg(lambda r: 1.0 if r.get("compression_pass", False) else 0.0)
        avg_S = avg(lambda r: r["soap_retention"].get("S",0.0))
        avg_O = avg(lambda r: r["soap_retention"].get("O",0.0))
        avg_A = avg(lambda r: r["soap_retention"].get("A",0.0))
        avg_P = avg(lambda r: r["soap_retention"].get("P",0.0))
        avg_dx = avg(lambda r: r["key_element_recall"].get("diagnoses",1.0))
        avg_labs = avg(lambda r: r["key_element_recall"].get("labs",1.0))
        avg_img = avg(lambda r: r["key_element_recall"].get("imaging",1.0))
        avg_comorb = avg(lambda r: r["key_element_recall"].get("comorbidities",1.0))
        avg_plan = avg(lambda r: r["key_element_recall"].get("plan_general",1.0))
        avg_adhf = avg(lambda r: r["key_element_recall"].get("plan_minimums_adhf",1.0))
        summary = {
            "avg_cases": n,
            "avg_compression_ratio": avg_compress,
            "avg_compression_pass_rate": avg_comp_pass,
            "avg_SOAP_retention": {"S": avg_S, "O": avg_O, "A": avg_A, "P": avg_P},
            "avg_key_element_recall": {
                "diagnoses": avg_dx, "labs": avg_labs, "imaging": avg_img,
                "comorbidities": avg_comorb, "plan_general": avg_plan, "plan_minimums_adhf": avg_adhf
            }
        }
        print("\n=== Macro averages ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
if __name__ == "__main__":
    main()
