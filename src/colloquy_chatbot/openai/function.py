def get_llm_metadata(func):
    function_data = getattr(func, "__llm_metadata__", None)
    return {
        "name": function_data["name"],
        "type": "function",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": function_data["parameters"],
        },
        "required": list(function_data["parameters"].keys()),
        "additionalProperties": False,
    }
