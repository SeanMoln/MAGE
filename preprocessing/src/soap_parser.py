"""
SOAP Section Parser for Clinical Notes
從臨床筆記中辨識並提取 SOAP (Subjective, Objective, Assessment, Plan) 區段
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class SOAPParser:
    """SOAP 區段解析器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化解析器

        Args:
            config_path: YAML 配置檔路徑
        """
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.keywords = config
        else:
            # 使用預設關鍵字
            self.keywords = self._get_default_keywords()

        self._compile_patterns()

    def _get_default_keywords(self) -> Dict:
        """預設 SOAP 關鍵字（含 Discharge Summary 常見標題）"""
        return {
            'subjective': {
                'primary': [
                    'subjective', 's:', 'chief complaint', 'cc:',
                    'history of present illness', 'history of the present illness',
                    'hpi', 'hpi:', 'presenting complaint', 'reason for hospitalization',
                    'reason for admission', 'presenting history',
                ],
                'secondary': ['patient reports', 'patient states', 'history:']
            },
            'objective': {
                'primary': [
                    'objective', 'o:', 'physical exam', 'physical examination',
                    'pe', 'pe:', 'vital signs', 'vitals', 'vitals:',
                    'laboratory data', 'lab data', 'laboratory results',
                    'pertinent results', 'radiologic data', 'radiology',
                    'hospital course', 'brief hospital course',
                    'summary of hospital course',
                ],
                'secondary': ['labs', 'laboratory', 'imaging', 'examination']
            },
            'assessment': {
                'primary': [
                    'assessment', 'a:', 'impression', 'diagnosis', 'dx:',
                    'assessment and plan', 'assessment/plan', 'a/p', 'a&p',
                    'discharge diagnosis', 'discharge diagnoses',
                    'final diagnosis', 'final diagnoses',
                    'principal diagnosis', 'primary diagnosis',
                    'secondary diagnoses', 'secondary diagnosis',
                    'problem list', 'active problems', 'active issues',
                ],
                'secondary': []
            },
            'plan': {
                'primary': [
                    'plan', 'p:', 'treatment plan', 'recommendations',
                    'discharge medications', 'discharge medication',
                    'medications on discharge', 'discharge instructions',
                    'discharge condition', 'discharge disposition',
                    'discharge plan', 'discharge status',
                    'follow-up', 'followup', 'follow up', 'follow up instructions',
                    'disposition', 'condition on discharge',
                ],
                'secondary': ['continue', 'start', 'follow-up', 'management']
            }
        }

    def _compile_patterns(self):
        """編譯正規表達式模式"""
        # 為每個 SOAP 區段建立模式
        self.section_patterns = {}

        for section in ['subjective', 'objective', 'assessment', 'plan']:
            keywords = self.keywords.get(section, {})
            primary = keywords.get('primary', [])

            # 建立模式：單詞邊界 + 關鍵字 + 可選冒號 + 換行或空白
            patterns = []
            for kw in primary:
                # 轉義特殊字符
                escaped_kw = re.escape(kw)
                # 匹配: 行首 + 關鍵字 + (冒號 或 行尾)
                # 允許標題後直接接內容（如 "PLAN: start..."），但不匹配句中出現
                pattern = rf'^\s*{escaped_kw}\s*(?::|$)'
                patterns.append(pattern)

            # 合併為單一模式
            combined_pattern = '|'.join(patterns)
            self.section_patterns[section] = re.compile(
                combined_pattern,
                re.MULTILINE | re.IGNORECASE
            )

    def parse(self, text: str, mode: str = 'hybrid') -> Dict[str, str]:
        """
        解析臨床筆記的 SOAP 區段

        Args:
            text: 清理後的臨床筆記文字
            mode: 解析模式 - 'rule_based', 'heuristic', 或 'hybrid'

        Returns:
            dict: {
                'S': subjective text,
                'O': objective text,
                'A': assessment text,
                'P': plan text,
                'parsing_method': 使用的解析方法,
                'confidence': 信心度 (0-1)
            }
        """
        if not text or not isinstance(text, str):
            return self._empty_result(error='Empty or invalid input')

        # 嘗試 rule-based 解析
        if mode in ['rule_based', 'hybrid']:
            result = self._rule_based_parse(text)
            if result['confidence'] > 0.5:
                return result

        # 如果 rule-based 失敗，嘗試 heuristic
        if mode in ['heuristic', 'hybrid']:
            result = self._heuristic_parse(text)
            return result

        return self._empty_result(error='Parsing failed')

    def _rule_based_parse(self, text: str) -> Dict[str, str]:
        """
        基於規則的解析：尋找 SOAP 標題關鍵字

        Args:
            text: 臨床筆記文字

        Returns:
            dict: SOAP 區段內容與信心度
        """
        lines = text.split('\n')
        sections = {'S': '', 'O': '', 'A': '', 'P': ''}
        current_section = None
        section_found = []

        section_map = {
            'subjective': 'S',
            'objective': 'O',
            'assessment': 'A',
            'plan': 'P'
        }

        for i, line in enumerate(lines):
            # 檢查是否為區段標題
            matched_section = None

            for section_key, pattern in self.section_patterns.items():
                if pattern.search(line):
                    matched_section = section_key
                    section_found.append(section_key)
                    break

            if matched_section:
                current_section = section_map[matched_section]
                # 提取標題同行的內嵌內容（如 "PLAN: start furosemide..."）
                inline_content = re.sub(
                    r'^\s*\S.*?:\s*', '', line, count=1, flags=re.IGNORECASE
                ).strip()
                if inline_content:
                    sections[current_section] += inline_content + '\n'
            elif current_section:
                # 收集當前區段的內容
                if line.strip():  # 忽略空行
                    sections[current_section] += line + '\n'

        # 清理各區段內容
        for key in sections:
            sections[key] = sections[key].strip()

        # 計算信心度
        confidence = len(section_found) / 4  # 找到幾個區段 / 4
        found_all_four = len(set(section_found)) == 4

        return {
            'S': sections['S'],
            'O': sections['O'],
            'A': sections['A'],
            'P': sections['P'],
            'parsing_method': 'rule_based',
            'confidence': confidence if found_all_four else confidence * 0.8,
            'sections_found': len(set(section_found))
        }

    def _heuristic_parse(self, text: str) -> Dict[str, str]:
        """
        基於啟發式的解析：根據內容特徵猜測 SOAP 區段

        Args:
            text: 臨床筆記文字

        Returns:
            dict: SOAP 區段內容與信心度
        """
        # 將文字分成段落
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if len(paragraphs) < 2:
            # 文字太短，無法有效分段
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        sections = {'S': [], 'O': [], 'A': [], 'P': []}

        # 啟發式指標
        subjective_indicators = ['complained', 'presented with', 'reported', 'pain', 'symptoms']
        objective_indicators = ['bp', 'hr', 'temp', 'exam', 'labs', 'x-ray', 'ct', 'mri']
        assessment_indicators = ['diagnosis', 'impression', 'likely', 'consistent with', 'suggests']
        plan_indicators = ['will', 'start', 'continue', 'discontinue', 'follow-up', 'discharge']

        for para in paragraphs:
            para_lower = para.lower()
            scores = {
                'S': sum(1 for ind in subjective_indicators if ind in para_lower),
                'O': sum(1 for ind in objective_indicators if ind in para_lower),
                'A': sum(1 for ind in assessment_indicators if ind in para_lower),
                'P': sum(1 for ind in plan_indicators if ind in para_lower)
            }

            # 分配到得分最高的區段
            max_section = max(scores, key=scores.get)
            if scores[max_section] > 0:
                sections[max_section].append(para)

        # 合併段落
        result = {
            'S': '\n\n'.join(sections['S']),
            'O': '\n\n'.join(sections['O']),
            'A': '\n\n'.join(sections['A']),
            'P': '\n\n'.join(sections['P']),
            'parsing_method': 'heuristic',
            'confidence': 0.4,  # Heuristic 信心度較低
            'sections_found': sum(1 for v in sections.values() if v)
        }

        return result

    def _empty_result(self, error: str = '') -> Dict[str, str]:
        """回傳空結果"""
        return {
            'S': '',
            'O': '',
            'A': '',
            'P': '',
            'parsing_method': 'none',
            'confidence': 0.0,
            'sections_found': 0,
            'error': error
        }

    def format_soap_text(self, sections: Dict[str, str],
                        include_empty: bool = False) -> str:
        """
        將 SOAP 區段格式化為單一字串

        Args:
            sections: SOAP 區段內容 dict
            include_empty: 是否包含空區段

        Returns:
            str: 格式化的 SOAP 文字，例如 "S: ... O: ... A: ... P: ..."
        """
        parts = []

        for section_key in ['S', 'O', 'A', 'P']:
            content = sections.get(section_key, '').strip()

            if content or include_empty:
                # 簡化內容：移除多餘換行
                content = re.sub(r'\n+', ' ', content)
                content = re.sub(r'\s+', ' ', content)

                if content:
                    parts.append(f"{section_key}: {content}")

        return ' '.join(parts)

    def batch_parse(self, texts: List[str], mode: str = 'hybrid',
                   show_progress: bool = True) -> List[Dict]:
        """
        批次解析多筆筆記

        Args:
            texts: 筆記文字列表
            mode: 解析模式
            show_progress: 是否顯示進度

        Returns:
            List[Dict]: 解析結果列表
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Parsing SOAP sections")
            except ImportError:
                logger.warning("tqdm not installed, progress bar disabled")
                iterator = texts
        else:
            iterator = texts

        for text in iterator:
            result = self.parse(text, mode=mode)
            results.append(result)

        # 統計
        successful = sum(1 for r in results if r['sections_found'] >= 2)
        avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0

        logger.info(f"Parsed {len(results)} texts")
        logger.info(f"Successful parses (≥2 sections): {successful} ({successful/len(results)*100:.1f}%)")
        logger.info(f"Average confidence: {avg_confidence:.2f}")

        return results


def main():
    """測試用主程式"""
    # 測試範例
    test_text = """
Subjective:
Patient is a 65-year-old male who presents with chest pain.
He reports the pain started 2 hours ago and radiates to left arm.

Objective:
Vital Signs: BP 140/90, HR 88, RR 16, Temp 98.6F
Physical Exam: Alert and oriented, no acute distress
Labs: Troponin negative, BNP elevated at 450
Chest X-ray: Clear lung fields

Assessment:
1. Acute decompensated heart failure
2. Hypertension - controlled
3. Chest pain - likely cardiac vs musculoskeletal

Plan:
- Admit to cardiology service
- Start IV furosemide 40mg
- Continue home medications
- Echocardiogram in AM
- Cardiology consult
- Monitor on telemetry
- Follow-up with PCP in 2 weeks
    """

    parser = SOAPParser()
    result = parser.parse(test_text, mode='hybrid')

    print("Parsing method:", result['parsing_method'])
    print("Confidence:", result['confidence'])
    print("Sections found:", result['sections_found'])
    print("\nSubjective:")
    print(result['S'])
    print("\nObjective:")
    print(result['O'])
    print("\nAssessment:")
    print(result['A'])
    print("\nPlan:")
    print(result['P'])

    # 測試格式化
    print("\n" + "="*50)
    print("Formatted SOAP text:")
    print("="*50)
    formatted = parser.format_soap_text(result)
    print(formatted)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
