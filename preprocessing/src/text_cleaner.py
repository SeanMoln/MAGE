"""
Text Cleaning Module for MIMIC Clinical Notes
清理臨床筆記：移除格式、處理去識別化標記、正規化文字
"""

import re
import logging
from typing import Dict, List, Optional
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class TextCleaner:
    """臨床筆記文字清理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化清理器

        Args:
            config_path: YAML 配置檔路徑，若為 None 則使用預設配置
        """
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.cleaning_config = config.get('cleaning', {})
                self.qc_config = config.get('quality_control', {}).get('thresholds', {})
        else:
            # 預設配置
            self.cleaning_config = {
                'target_encoding': 'utf-8',
                'normalize_linebreaks': True,
                'max_consecutive_linebreaks': 2,
                'normalize_whitespace': True,
                'trim_lines': True,
            }
            self.qc_config = {
                'min_words': 50,
                'max_words': 10000,
            }

        self._compile_patterns()

    def _compile_patterns(self):
        """預編譯正規表達式模式以提升效能"""

        # 去識別化標記
        self.deid_pattern = re.compile(r'\[\*\*[^\]]*\*\*\]', re.IGNORECASE)

        # 裝飾性符號
        self.decorative_patterns = [
            re.compile(r'^[=]{5,}$', re.MULTILINE),
            re.compile(r'^[-]{5,}$', re.MULTILINE),
            re.compile(r'^[*]{5,}$', re.MULTILINE),
            re.compile(r'^[_]{5,}$', re.MULTILINE),
        ]

        # 頁眉頁腳
        self.header_footer_patterns = [
            re.compile(r'^Page \d+ of \d+.*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^Printed on.*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^Confidential.*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^MRN:.*$', re.MULTILINE),
            re.compile(r'^Medical Record Number:.*$', re.MULTILINE | re.IGNORECASE),
        ]

        # 多餘空白
        self.multiple_spaces = re.compile(r' {2,}')
        self.multiple_newlines = re.compile(r'\n{3,}')

    def clean_text(self, text: str) -> Dict[str, any]:
        """
        清理單一筆記

        Args:
            text: 原始臨床筆記文字

        Returns:
            dict: {
                'cleaned_text': 清理後文字,
                'original_length': 原始長度,
                'cleaned_length': 清理後長度,
                'removed_deid_count': 移除的去識別化標記數量,
                'is_valid': 是否通過基本品質檢查
            }
        """
        if not text or not isinstance(text, str):
            return {
                'cleaned_text': '',
                'original_length': 0,
                'cleaned_length': 0,
                'removed_deid_count': 0,
                'is_valid': False,
                'error': 'Empty or invalid input'
            }

        original_length = len(text)
        cleaned = text

        # Step 1: 處理去識別化標記
        deid_matches = self.deid_pattern.findall(cleaned)
        deid_count = len(deid_matches)

        # 替換為簡化標記或完全移除
        if self.cleaning_config.get('deidentification', {}).get('action') == 'replace':
            cleaned = self._replace_deid_markers(cleaned)
        else:
            cleaned = self.deid_pattern.sub('', cleaned)

        # Step 2: 移除裝飾性符號
        for pattern in self.decorative_patterns:
            cleaned = pattern.sub('', cleaned)

        # Step 3: 移除頁眉頁腳
        for pattern in self.header_footer_patterns:
            cleaned = pattern.sub('', cleaned)

        # Step 4: 正規化空白與換行
        if self.cleaning_config.get('normalize_whitespace'):
            # 移除行尾空白
            if self.cleaning_config.get('trim_lines'):
                lines = [line.rstrip() for line in cleaned.split('\n')]
                cleaned = '\n'.join(lines)

            # 統一多重空格為單一空格
            cleaned = self.multiple_spaces.sub(' ', cleaned)

        if self.cleaning_config.get('normalize_linebreaks'):
            # 限制連續換行數量
            max_newlines = self.cleaning_config.get('max_consecutive_linebreaks', 2)
            pattern = re.compile(r'\n{' + str(max_newlines + 1) + ',}')
            cleaned = pattern.sub('\n' * max_newlines, cleaned)

        # Step 5: 移除前後空白
        cleaned = cleaned.strip()

        # Step 6: 品質檢查
        word_count = len(cleaned.split())
        is_valid = self._check_quality(cleaned, word_count)

        cleaned_length = len(cleaned)

        return {
            'cleaned_text': cleaned,
            'original_length': original_length,
            'cleaned_length': cleaned_length,
            'word_count': word_count,
            'removed_deid_count': deid_count,
            'compression_ratio': cleaned_length / original_length if original_length > 0 else 0,
            'is_valid': is_valid,
        }

    def _replace_deid_markers(self, text: str) -> str:
        """
        將去識別化標記替換為簡化標籤

        Examples:
            [**Name (NI) 123**] -> <NAME>
            [**Hospital1 456**] -> <HOSPITAL>
            [**2024-1-1**] -> <DATE>
        """

        def replace_marker(match):
            marker = match.group(0).lower()

            if 'name' in marker:
                return '<NAME>'
            elif 'hospital' in marker or 'location' in marker:
                return '<HOSPITAL>'
            elif 'date' in marker or any(char.isdigit() and '-' in marker for char in marker):
                return '<DATE>'
            elif 'age' in marker:
                return '<AGE>'
            elif 'identifier' in marker or 'id' in marker or 'number' in marker:
                return '<ID>'
            else:
                return '<REDACTED>'

        return self.deid_pattern.sub(replace_marker, text)

    def _check_quality(self, text: str, word_count: int) -> bool:
        """
        檢查清理後文字是否符合品質標準

        Args:
            text: 清理後文字
            word_count: 字數

        Returns:
            bool: 是否通過檢查
        """
        min_words = self.qc_config.get('min_words', 50)
        max_words = self.qc_config.get('max_words', 10000)

        # 檢查字數範圍
        if word_count < min_words:
            logger.debug(f"Text too short: {word_count} words (min: {min_words})")
            return False

        if word_count > max_words:
            logger.debug(f"Text too long: {word_count} words (max: {max_words})")
            return False

        # 檢查是否過於重複
        if self._is_too_repetitive(text):
            logger.debug("Text is too repetitive")
            return False

        return True

    def _is_too_repetitive(self, text: str, threshold: float = 0.8) -> bool:
        """
        檢查文字是否過度重複（可能是錯誤或模板）

        Args:
            text: 文字內容
            threshold: 重複比例門檻

        Returns:
            bool: 是否過度重複
        """
        lines = text.split('\n')
        if len(lines) < 5:
            return False

        # 計算獨特行的比例
        unique_lines = len(set(lines))
        repetition_ratio = 1 - (unique_lines / len(lines))

        return repetition_ratio > threshold

    def batch_clean(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        """
        批次清理多筆筆記

        Args:
            texts: 筆記文字列表
            show_progress: 是否顯示進度

        Returns:
            List[Dict]: 清理結果列表
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Cleaning texts")
            except ImportError:
                logger.warning("tqdm not installed, progress bar disabled")
                iterator = texts
        else:
            iterator = texts

        for text in iterator:
            result = self.clean_text(text)
            results.append(result)

        # 統計資訊
        valid_count = sum(1 for r in results if r['is_valid'])
        logger.info(f"Cleaned {len(results)} texts, {valid_count} valid ({valid_count/len(results)*100:.1f}%)")

        return results


def main():
    """測試用主程式"""
    # 測試範例
    test_text = """
Page 1 of 2                    [**Hospital1 123**]

=====================================================
                 DISCHARGE SUMMARY
=====================================================

Patient Name: [**Name (NI) 456**]
DOB: [**2024-1-1**]


Subjective:
Patient is a 65 year old male who presented with chest pain.
He reports the pain started 2 hours ago.


Objective:
BP 140/90, HR 88, RR 16, Temp 98.6F
Chest X-ray shows clear lung fields


Assessment:
1. Chest pain - likely musculoskeletal
2. Hypertension


Plan:
- Continue home medications
- Follow-up in 1 week
- Return if symptoms worsen


=========================
Printed on 2024-11-18
    """

    cleaner = TextCleaner()
    result = cleaner.clean_text(test_text)

    print("Original length:", result['original_length'])
    print("Cleaned length:", result['cleaned_length'])
    print("Word count:", result['word_count'])
    print("Removed deidentification markers:", result['removed_deid_count'])
    print("Is valid:", result['is_valid'])
    print("\nCleaned text:")
    print(result['cleaned_text'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
