import subprocess
import re
from typing import Optional

def get_current_run_id(app_name: str) -> Optional[str]:
    """
    執行 flwr run --list 並解析出最新的 RUN_ID。
    假設最新的 Run 會出現在列表的第一行或具有特定狀態。
    """
    try:

        # Change directory to where the flwr command is available if necessary
        if app_name == "NES":
            work_dir = f"/app/app/network-energy-saving"
        else:
            work_dir = f"/app/app/{app_name}"

        # 執行指令並取得輸出內容
        result = subprocess.run(
            ["flwr", "list", "--runs"], 
            cwd=work_dir,
            capture_output=True, 
            text=True, 
            check=True
        )
        
        output = result.stdout
        
        # 典型的 flwr run --list 輸出格式通常包含表格
        # 我們尋找看起來像數字或特定 ID 格式的字串
        # 這裡假設 ID 是純數字，且我們取第一筆找到的資料
        lines = output.strip().split('\n')
        print(lines)
        
        if len(lines) < 6:
            print("目前沒有偵測到任何運行中的 Run。")
            return None

        # 簡單解析邏輯：跳過表頭，尋找第一行數據中的第一個欄位
        # 這裡使用正則表達式尋找每一行開頭的數字 (RUN_ID)
        for line in lines:
            match = re.search(r'\d{10,}', line)
            if match:
                current_run_id = match.group(0)
                return current_run_id
                
        return None

    except subprocess.CalledProcessError as e:
        print(f"執行指令失敗: {e}")
        return None
    except Exception as e:
        print(f"發生錯誤: {e}")
        return None
    
if __name__ == "__main__":
    app_name = "NES"  # 替換為你的應用名稱
    run_id = get_current_run_id(app_name)
    if run_id:
        print(f"目前的 RUN_ID 是: {run_id}")
    else:
        print("未找到有效的 RUN_ID。")