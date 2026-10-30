"""A scripted chat model: a handler decides the reply from (system_prompt, user_prompt)."""
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    handler: Any
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = next((m.content for m in messages if m.type == "system"), "")
        user = next((m.content for m in reversed(messages) if m.type == "human"), "")
        self.calls.append((system[:30], user[:60]))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.handler(system, user)))])


def make_handler(router=None, validator=None, rag="From docs [1].", web="From web [1].", llm="From model."):
    """router/validator: lists consumed in order (last item repeats); others: fixed answers."""
    router, validator = list(router or []), list(validator or [])

    def pop(q):
        return q.pop(0) if len(q) > 1 else q[0]

    def handler(system, user):
        if system.startswith("You are the supervisor"):
            return pop(router)
        if system.startswith("You are the validator"):
            return pop(validator)
        if system.startswith("You answer questions using ONLY the numbered document"):
            return rag
        if system.startswith("You answer questions using ONLY the numbered web"):
            return web
        return llm

    return handler
