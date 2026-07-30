import json
import os
from pathlib import Path

# 以 src/utils/ 為基準，往上兩層到專案根目錄
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_JSON  = _PROJECT_ROOT / "data" / "memory" / "prompt_memory.json"
_DEFAULT_EXCEL = _PROJECT_ROOT / "data" / "memory" / "prompt_memory.xlsx"

def reset_prompt_memory(json_path=None, excel_path=None):
    if json_path is None:
        json_path = _DEFAULT_JSON
    if excel_path is None:
        excel_path = _DEFAULT_EXCEL

    json_path  = Path(json_path)
    excel_path = Path(excel_path)

    # 覆蓋 JSON 檔案
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)
    print(f"[Reset] {json_path} 已重置為空。")

    # 如果有 Excel 檔案，也一併清空
    if excel_path.exists():
        try:
            import pandas as pd
            pd.DataFrame([]).to_excel(excel_path, index=False)
            print(f"[Reset] {excel_path} 已重置為空。")
        except Exception as e:
            print(f"[Warning] Excel 清空失敗: {e}")

if __name__ == "__main__":
    reset_prompt_memory()
