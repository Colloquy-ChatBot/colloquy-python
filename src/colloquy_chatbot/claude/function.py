def get_llm_metadata(func):
    function_data = getattr(func, "__llm_metadata__", None)
    if function_data:
        return {
            "name": func.__name__,
            "description": function_data["description"],
            "input_schema": {
                "type": "object",
                "properties": function_data["parameters"],
            },
        }
