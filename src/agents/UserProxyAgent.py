class UserProxyAgent:
    def __init__(self):
        pass

    def receive_input(self, task_description, input_text):
        print("\n[Progress] Receiving input from user...\n")
        # 模擬接收用戶輸入的任務描述
        # 在實際應用中，這裡可以是從文件、API 或其他來源獲取數據
        if not task_description:
            raise ValueError("Task description cannot be empty.")
        return {"task_description": task_description, "base_prompt": "Optimize this prompt for the task:","input_text":input_text}
