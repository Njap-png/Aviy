from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

MAX_MESSAGE_LENGTH = 1_000_000
MAX_TOOL_ARGUMENT_BYTES = 1_000_000
MAX_TOOL_RESULT_BYTES = 1_000_000


def _json_copy(value: object, *, maximum_bytes: int | None = None) -> object:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Value must contain finite JSON-compatible data.") from exc

    if maximum_bytes is not None and len(encoded) > maximum_bytes:
        raise ValueError("Serialized value exceeds the configured size limit.")

    return json.loads(encoded.decode("utf-8"))


@dataclass(frozen=True)
class AgentMessage:
    sender: str
    recipient: str
    content: str
    created_at: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sender, str)
            or not self.sender.strip()
            or not isinstance(self.recipient, str)
            or not self.recipient.strip()
        ):
            raise ValueError("Message sender and recipient are required.")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Message content cannot be empty.")
        if len(self.content) > MAX_MESSAGE_LENGTH:
            raise ValueError("Message exceeds the maximum permitted length.")
        if (
            isinstance(self.created_at, bool)
            or not isinstance(self.created_at, (int, float))
            or not math.isfinite(self.created_at)
            or self.created_at <= 0
        ):
            raise ValueError("Message timestamp must be finite and positive.")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    handler: Callable[[dict[str, object]], dict[str, object]]
    required_permission: str
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tool name is required.")
        if not isinstance(self.required_permission, str) or not self.required_permission.strip():
            raise ValueError("Tool permission is required.")
        if not callable(self.handler):
            raise TypeError("Tool handler must be callable.")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("Tool timeout must be finite and positive.")


class OSIsolationBackend(Protocol):
    @property
    def isolation_verified(self) -> bool: ...

    def execute(
        self,
        tool: ToolDefinition,
        arguments: dict[str, object],
        *,
        timeout_seconds: float,
    ) -> dict[str, object]: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._lock = threading.RLock()

    def register(self, tool: ToolDefinition) -> None:
        if not isinstance(tool, ToolDefinition):
            raise TypeError("Only ToolDefinition instances can be registered.")
        with self._lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool already registered: {tool.name}")
            self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        with self._lock:
            try:
                return self._tools[name]
            except KeyError as exc:
                raise KeyError(f"Unknown tool: {name}") from exc

    @property
    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._tools))


class AuditTrail:
    """Tamper-evident in-memory hash chain; not a signed remote audit service."""

    def __init__(self) -> None:
        self._events: list[dict[str, object]] = []
        self._last_hash = "0" * 64
        self._lock = threading.RLock()

    @property
    def events(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(copy.deepcopy(event) for event in self._events)

    def record(self, event: str, details: dict[str, object]) -> str:
        if not isinstance(event, str) or not event.strip():
            raise ValueError("Audit event name is required.")
        if not isinstance(details, dict):
            raise TypeError("Audit details must be a dictionary.")

        safe_details = _json_copy(details)
        with self._lock:
            body = {
                "sequence": len(self._events),
                "timestamp": time.time(),
                "event": event,
                "details": safe_details,
                "previous_hash": self._last_hash,
            }
            encoded = json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            self._events.append({**body, "hash": digest})
            self._last_hash = digest
            return digest

    def verify(self) -> bool:
        with self._lock:
            previous = "0" * 64
            for index, event in enumerate(self._events):
                if not isinstance(event, dict):
                    return False
                body = {
                    key: value for key, value in event.items()
                    if key != "hash"
                }
                if body.get("sequence") != index:
                    return False
                if body.get("previous_hash") != previous:
                    return False
                try:
                    encoded = json.dumps(
                        body,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                except (TypeError, ValueError):
                    return False
                digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                if digest != event.get("hash"):
                    return False
                previous = digest
            return previous == self._last_hash

    def export_json(self) -> str:
        with self._lock:
            if not self.verify():
                raise RuntimeError("Audit trail integrity verification failed.")
            return json.dumps(self._events, sort_keys=True, indent=2)


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    assigned_agent: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("Task ID is required.")
        if not isinstance(self.assigned_agent, str) or not self.assigned_agent.strip():
            raise ValueError("Assigned agent is required.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Task description is required.")


class TaskCoordinator:
    """Tracks task assignment and explicit, validated lifecycle transitions."""

    _TRANSITIONS = MappingProxyType({
        "queued": frozenset({"running", "cancelled"}),
        "running": frozenset({"completed", "failed", "cancelled"}),
        "completed": frozenset(),
        "failed": frozenset(),
        "cancelled": frozenset(),
    })

    def __init__(self, audit: AuditTrail | None = None) -> None:
        self._tasks: dict[str, AgentTask] = {}
        self._statuses: dict[str, str] = {}
        self._audit = audit
        self._lock = threading.RLock()

    def assign(self, task: AgentTask) -> None:
        if not isinstance(task, AgentTask):
            raise TypeError("Only AgentTask instances can be assigned.")
        with self._lock:
            if task.task_id in self._tasks:
                raise ValueError("Task ID already exists.")
            if self._audit is not None:
                self._audit.record("task_assigned", {
                    "task_id": task.task_id,
                    "agent": task.assigned_agent,
                })
            self._tasks[task.task_id] = task
            self._statuses[task.task_id] = "queued"

    def get(self, task_id: str) -> AgentTask:
        with self._lock:
            return self._tasks[task_id]

    def status(self, task_id: str) -> str:
        with self._lock:
            return self._statuses[task_id]

    def transition(self, task_id: str, new_status: str) -> None:
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Unknown task: {task_id}")
            current = self._statuses[task_id]
            if new_status not in self._TRANSITIONS.get(current, set()):
                raise ValueError(
                    f"Invalid task transition: {current} -> {new_status}"
                )
            if self._audit is not None:
                self._audit.record("task_status_changed", {
                    "task_id": task_id,
                    "previous_status": current,
                    "new_status": new_status,
                })
            self._statuses[task_id] = new_status

    @property
    def tasks(self) -> tuple[AgentTask, ...]:
        with self._lock:
            return tuple(self._tasks.values())


class AgentRuntime:
    def __init__(
        self,
        agent_id: str,
        permissions: set[str],
        tools: ToolRegistry,
        audit: AuditTrail,
        executor: OSIsolationBackend | None = None,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("Agent ID is required.")
        if not isinstance(permissions, (set, frozenset)):
            raise TypeError("Permissions must be a set of strings.")
        if any(not isinstance(item, str) or not item.strip() for item in permissions):
            raise ValueError("Permissions must be non-empty strings.")
        if not isinstance(tools, ToolRegistry) or not isinstance(audit, AuditTrail):
            raise TypeError("A ToolRegistry and AuditTrail are required.")

        self.agent_id = agent_id
        self.permissions = frozenset(permissions)
        self.tools = tools
        self.audit = audit
        self.executor = executor
        self._inbox: list[AgentMessage] = []
        self._active = False
        self._lock = threading.RLock()

    @property
    def state(self) -> str:
        with self._lock:
            return "active" if self._active else "stopped"

    def start(self) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError("Agent is already active.")
            self.audit.record("agent_started", {"agent": self.agent_id})
            self._active = True

    def stop(self) -> None:
        with self._lock:
            if not self._active:
                raise RuntimeError("Agent is not active.")
            self.audit.record("agent_stopped", {"agent": self.agent_id})
            self._active = False

    def send(self, recipient: AgentRuntime, content: str) -> None:
        if not isinstance(recipient, AgentRuntime):
            raise TypeError("Recipient must be an AgentRuntime.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message cannot be empty.")
        if len(content) > MAX_MESSAGE_LENGTH:
            raise ValueError("Message exceeds the maximum permitted length.")

        # Lock agents in a stable order to avoid cross-send deadlocks.
        first, second = sorted(
            (self, recipient), key=lambda item: (item.agent_id, id(item))
        )
        with first._lock, second._lock:
                if not self._active or not recipient._active:
                    raise RuntimeError("Both agents must be active.")
                message = AgentMessage(
                    self.agent_id, recipient.agent_id, content, time.time()
                )
                self.audit.record("message_sent", {
                    "sender": self.agent_id,
                    "recipient": recipient.agent_id,
                })
                recipient._inbox.append(message)

    def receive(self) -> tuple[AgentMessage, ...]:
        with self._lock:
            if not self._active:
                raise RuntimeError("Agent is not active.")
            messages = tuple(self._inbox)
            self.audit.record("messages_received", {
                "agent": self.agent_id,
                "count": len(messages),
            })
            self._inbox.clear()
            return messages

    def invoke(
        self, tool_name: str, arguments: dict[str, object]
    ) -> dict[str, object]:
        with self._lock:
            if not self._active:
                raise RuntimeError("Agent is not active.")

        tool = self.tools.get(tool_name)

        if tool.required_permission not in self.permissions:
            self.audit.record("tool_denied", {
                "agent": self.agent_id,
                "tool": tool_name,
            })
            raise PermissionError("Agent lacks the required tool permission.")

        if self.executor is None:
            self.audit.record("tool_blocked_no_isolation", {
                "agent": self.agent_id,
                "tool": tool_name,
            })
            raise RuntimeError(
                "Tool execution blocked: a verified OS-isolation backend is required."
            )

        try:
            isolation_verified = self.executor.isolation_verified
        except Exception as exc:
            self.audit.record("tool_blocked_isolation_error", {
                "agent": self.agent_id,
                "tool": tool_name,
                "error_type": type(exc).__name__,
            })
            raise RuntimeError("Isolation backend verification failed.") from exc

        if isolation_verified is not True:
            self.audit.record("tool_blocked_no_isolation", {
                "agent": self.agent_id,
                "tool": tool_name,
            })
            raise RuntimeError(
                "Tool execution blocked: a verified OS-isolation backend is required."
            )

        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a dictionary.")

        safe_arguments = _json_copy(
            arguments, maximum_bytes=MAX_TOOL_ARGUMENT_BYTES
        )
        self.audit.record("tool_started", {
            "agent": self.agent_id,
            "tool": tool_name,
        })

        try:
            result = self.executor.execute(
                tool,
                safe_arguments,
                timeout_seconds=tool.timeout_seconds,
            )
            if not isinstance(result, dict):
                raise TypeError("Isolated tools must return a dictionary.")
            safe_result = _json_copy(
                result, maximum_bytes=MAX_TOOL_RESULT_BYTES
            )
        except Exception as exc:
            self.audit.record("tool_failed", {
                "agent": self.agent_id,
                "tool": tool_name,
                "error_type": type(exc).__name__,
            })
            raise

        self.audit.record("tool_completed", {
            "agent": self.agent_id,
            "tool": tool_name,
        })
        return safe_result
