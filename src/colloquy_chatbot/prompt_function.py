"""Function descriptor for LLM function calling."""

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Type, get_type_hints


class prompt_function:
    def __init__(
        self,
        description: Optional[str] = None,
        name: Optional[str] = None,
        parameter_descriptions: Optional[Dict[str, str]] = None,
    ):
        self.description = description
        self.name = name
        self.parameter_descriptions = parameter_descriptions or {}
        self.function = None
        self.metadata: Dict[str, Any] = {}

    def __call__(self, func):
        self.function = func
        self.name = self.name or func.__name__
        self.description = self.description or (func.__doc__ or "").strip()
        self.metadata = {
            "name": self.name,
            "description": self.description,
            "parameters": self._infer_parameters(func),
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__llm_function__ = True
        wrapper.__llm_metadata__ = self.metadata
        return wrapper

    def _infer_parameters(self, function: Callable) -> Dict[str, Dict[str, Any]]:
        signature = inspect.signature(function)
        type_hints = get_type_hints(function)
        params = {}

        for name, param in signature.parameters.items():
            param_info = {}

            # Get description from provided parameter descriptions or default
            param_info["description"] = self.parameter_descriptions.get(
                name, ""
            )

            # Infer type from type hints or default value
            if name in type_hints:
                param_type = type_hints[name]
                param_info["type"] = self._python_type_to_json_type(param_type)
            elif param.default is not inspect.Parameter.empty:
                param_info["type"] = self._python_type_to_json_type(type(param.default))

            params[name] = param_info

        return params

    def _python_type_to_json_type(self, py_type: Type) -> str:
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        # Handle Optional types
        origin = getattr(py_type, "__origin__", None)
        if origin is not None:
            args = getattr(py_type, "__args__", [])
            if len(args) == 2 and type(None) in args:
                # This is Optional[X]
                for arg in args:
                    if arg is not type(None):
                        return self._python_type_to_json_type(arg)

        return type_map.get(py_type, "string")


def get_llm_functions(obj):
    functions = []
    for name in dir(obj):
        attr = getattr(obj, name)
        if callable(attr) and getattr(attr, "__llm_function__", False):
            functions.append((attr, getattr(attr, "__llm_metadata__")))
    return functions

def catalog_functions(functions):
    return {
        getattr(function, "__llm_metadata__", {})["name"]: function
        for function in functions
    }
