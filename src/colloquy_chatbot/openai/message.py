from .. import message as base
import json

class RoleMessage(base.RoleMessage):
    def input(self):
        return {
            "role": self.role,
            "content": self.text,
        }

class FunctionCallMessage(base.FunctionCallMessage):
    def __init__(self, fn, content):
        super().__init__(fn, content.call_id, json.loads(content.arguments))
        self["content"] = self.content = content
        self.result_class = FunctionResultMessage

    def input(self):
        return self.content

class FunctionResultMessage(base.FunctionResultMessage):
    def input(self):
        return {
            "type": "function_call_output",
            "call_id": self.id,
            "output": self.result,
        }
