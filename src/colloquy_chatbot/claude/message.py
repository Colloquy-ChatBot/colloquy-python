from .. import message as base

class RoleMessage(base.RoleMessage):
    def input(self):
        return {
            "role": self.role,
            "content": self.text,
        }


class FunctionCallMessage(base.FunctionCallMessage):
    def __init__(self, function, content):
        super().__init__(function, content.id, content.input)
        self["content"] = self.content = content
        self.result_class = FunctionResultMessage

    def input(self):
        return {
            "role": "assistant",
            "content": [self.content],
        }


class FunctionResultMessage(base.FunctionResultMessage):
    def input(self):
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": self.id,
                "content": self.result,
            }],
        }
