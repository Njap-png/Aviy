from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AgentMessage:
    sender: str
    recipient: str
    content: str
    created_at: float


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    handler: Callable[[dict[str, object]], dict[str, object]]
    required_permission: str


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if not tool.name.strip() or not tool.required_permission.strip():
            raise ValueError("Tool name and permission are required.")
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc


class AuditTrail:
    def __init__(self) -> None:
        self._events: list[dict[str, object]] = []
        self._last_hash = "0" * 64

    @property
    def events(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(event) for event in self._events)

    def record(self, event: str, details: dict[str, object]) -> str:
        body = {
            "sequence": len(self._events),
            "event": event,
            "details": details,
            "previous_hash": self._last_hash,
        }
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self._events.append({**body, "hash": digest})
        self._last_hash = digest
        return digest


class AgentRuntime:
    def __init__(
        self,
        agent_id: str,
        permissions: set[str],
        tools: ToolRegistry,
        audit: AuditTrail,
    ) -> None:
        if not agent_id.strip():
            raise ValueError("Agent ID is required.")
        self.agent_id = agent_id
        self.permissions = frozenset(permissions)
        self.tools = tools
        self.audit = audit
        self._inbox: list[AgentMessage] = []
        self._active = False

    def start(self) -> None:
        if self._active:
            raise RuntimeError("Agent is already active.")
        self._active = True
        self.audit.record("agent_started", {"agent": self.agent_id})

    def stop(self) -> None:
        self._active = False
        self.audit.record("agent_stopped", {"agent": self.agent_id})

    def send(self, recipient: AgentRuntime, content: str) -> None:
        if not self._active or not recipient._active:
            raise RuntimeError("Both agents must be active.")
        if not content.strip():
            raise ValueError("Message cannot be empty.")
        message = AgentMessage(
            self.agent_id, recipient.agent_id, content, time.time()
        )
        recipient._inbox.append(message)
        self.audit.record(
            "message_sent",
            {"sender": self.agent_id, "recipient": recipient.agent_id},
        )

    def receive(self) -> tuple[AgentMessage, ...]:
        if not self._active:
            raise RuntimeError("Agent is not active.")
        messages = tuple(self._inbox)
        self._inbox.clear()
        return messages

    def invoke(
        self, tool_name: str, arguments: dict[str, object]
    ) -> dict[str, object]:
        if not self._active:
            raise RuntimeError("Agent is not active.")
        tool = self.tools.get(tool_name)
        if tool.required_permission not in self.permissions:
            self.audit.record(
                "tool_denied",
                {"agent": self.agent_id, "tool": tool_name},
            )
            raise PermissionError("Agent lacks the required tool permission.")

        # JSON round-trip prevents callers and handlers sharing mutable objects.
        isolated_arguments = json.loads(json.dumps(arguments))
        self.audit.record(
            "tool_started", {"agent": self.agent_id, "tool": tool_name}
        )
        result = tool.handler(isolated_arguments)
        if not isinstance(result, dict):
            raise TypeError("Tool handlers must return a dictionary.")
        isolated_result = json.loads(json.dumps(result))
        self.audit.record(
            "tool_completed", {"agent": self.agent_id, "tool": tool_name}
        )
        return isolated_result
