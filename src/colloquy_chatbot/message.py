# Inherits from dict to make equality easier
class Message[T](dict):
    def input(self) -> T:
        raise NotImplementedError(self)

class TextMessage(Message):
    def __init__(self, text: str):
        self["text"] = self.text = text

class RoleMessage[Role](TextMessage):
    def __init__(self, role: Role, text: str):
        super().__init__(text)
        self["role"] = self.role = role


class FunctionCallMessage(Message):
    def __init__(self, function, id, kwargs):
        self["function"] = self.function = function
        self["id"] = self.id = id
        self["kwargs"] = self.kwargs = kwargs

    def invoke(self):
        return self.result_class(
            self.id,
            self.function(**self.kwargs),
        )


class FunctionResultMessage(Message):
    def __init__(self, id, result):
        self["id"] = self.id = id
        self["result"] = self.result = result
