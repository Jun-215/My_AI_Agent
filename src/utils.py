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