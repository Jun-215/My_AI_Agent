# main.py (部分)
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.core import Agent
from src.tools import get_current_datetime, add, compare, count_letter_in_string

load_dotenv()
API_KEY = os.getenv('SF_API_KEY')


if __name__ == "__main__":
    client = OpenAI(
        api_key=API_KEY, # 替换为你的 API Key
        base_url="https://api.siliconflow.cn/v1",
        
    )

    # 创建 Agent 实例，传入 client、模型名称和工具函数列表
    agent = Agent(
        client=client,
        model="Qwen/QwQ-32B",
        tools=[get_current_datetime, add, compare, count_letter_in_string],
        verbose=True # 设置为 True 可以看到工具调用信息
    )

    # 开始交互式对话循环
    while True:
        # 使用彩色输出区分用户输入和AI回答
        prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示
        if prompt.lower() == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response,"\n")  # 绿色显示AI助手回答