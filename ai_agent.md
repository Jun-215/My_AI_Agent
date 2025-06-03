动手造一个Tiny-Agent

我们来基于 库和其 功能，动手构造一个 Tiny-Agent，这个 Agent 是一个简单的任务导向型 Agent，它能够根据用户的输入，回答一些简单的问题。



最终的效果：

![74893646187](C:\Users\Jun\AppData\Local\Temp\1748936461876.png)



第一步：

先获取到AI agent的地基模型api，没注册需要先注册

[网址：SiliconFlow](https://www.siliconflow.cn/)

![74893887173](C:\Users\Jun\AppData\Local\Temp\1748938871731.png)



![74893893058](C:\Users\Jun\AppData\Local\Temp\1748938930582.png)



![74893630858](C:\Users\Jun\AppData\Local\Temp\1748936308585.png)



![74893637683](C:\Users\Jun\AppData\Local\Temp\1748936376833.png)



第二步：

项目的目录结构：

![74893676493](C:\Users\Jun\AppData\Local\Temp\1748936764933.png)







定义工具函数：

```
# src/tools.py
from datetime import datetime

# 获取当前日期和时间
def get_current_datetime() -> str:
    """
    获取当前日期和时间。
    :return: 当前日期和时间的字符串表示。
    """
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime

def add(a: float, b: float):
    """
    计算两个浮点数的和。
    :param a: 第一个浮点数。
    :param b: 第二个浮点数。
    :return: 两个浮点数的和。
    """
    return str(a + b)

def compare(a: float, b: float):
    """
    比较两个浮点数的大小。
    :param a: 第一个浮点数。
    :param b: 第二个浮点数。
    :return: 比较结果的字符串表示。
    """
    if a > b:
        return f'{a} is greater than {b}'
    elif a < b:
        return f'{b} is greater than {a}'
    else:
        return f'{a} is equal to {b}'

def count_letter_in_string(a: str, b: str):
    """
    统计字符串中某个字母的出现次数。
    :param a: 要搜索的字符串。
    :param b: 要统计的字母。
    :return: 字母在字符串中出现的次数。
    """
    return str(a.count(b))

# ... (可能还有其他工具函数)
```

```
# src/utils.py (部分)
import inspect

def function_to_json(func) -> dict:
    sig = inspect.signature(func)
    parameters = {}
    required = []
    for name, param in sig.parameters.items():
        # 根據型別推斷 OpenAI schema
        if param.annotation == float:
            param_type = "number"
        elif param.annotation == int:
            param_type = "integer"
        elif param.annotation == str:
            param_type = "string"
        else:
            param_type = "string"
        parameters[name] = {"type": param_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
```



构造agent类：

```
# src/core.py (部分)
from openai import OpenAI
import json
from typing import List, Dict, Any
from src.utils import function_to_json
# 导入定义好的工具函数
from src.tools import get_current_datetime, add, compare, count_letter_in_string

SYSREM_PROMPT = """
你是一个叫JUN人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""

class Agent:
    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen2.5-32B-Instruct", tools: List=[], verbose : bool = True):
        self.client = client
        self.tools = tools
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSREM_PROMPT},
        ]
        self.verbose = verbose

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # 获取所有工具的 JSON 模式
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # 处理工具调用
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        function_call_content = eval(f"{function_name}(**{function_args})")

        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:

        self.messages.append({"role": "user", "content": prompt})

        # 获取模型的完成响应
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )
        # print("DEBUG response:", response)
        if isinstance(response, str):
            # print("API 返回字串，內容為：", response)
            return response
        # 检查模型是否调用了工具        
        if response.choices[0].message.tool_calls:
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            # 处理工具调用
            tool_list = []
            for tool_call in response.choices[0].message.tool_calls:
                # 处理工具调用并将结果添加到消息列表中
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append([tool_call.function.name, tool_call.function.arguments])
            # 调用过程
            # if self.verbose:
            #     print("调用工具：", response.choices[0].message.content, tool_list)
            # 再次获取模型的完成响应，这次包含工具调用的结果
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )
            if isinstance(response, str):
                # print("API 返回字串，內容為：", response)
                return response

        # 将模型的完成响应添加到消息列表中
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
```



主函数：

```
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
```

```
# .env 部分
SF_API_KEY = 你在平台获取的api_key
```



Agent 的工作流程如下：

1. 接收用户输入。
2. 调用大模型（如 Qwen），并告知其可用的工具及其 Schema。
3. 如果模型决定调用工具，Agent 会解析请求，执行相应的 Python 函数。
4. Agent 将工具的执行结果返回给模型。
5. 模型根据工具结果生成最终回复。
6. Agent 将最终回复返回给用户。

![图7.9 Agent 工作流程](https://datawhalechina.github.io/happy-llm/images/7-images/7-3-Tiny_Agent.jpg)