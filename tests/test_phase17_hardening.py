import math

import pytest

from aviy.agents.runtime import (
    AgentMessage,
    AgentRuntime,
    AgentTask,
    AuditTrail,
    TaskCoordinator,
    ToolDefinition,
    ToolRegistry,
)


class RecordingExecutor:
    """Test double for orchestration only; not a real security sandbox."""

    isolation_verified = True

    def __init__(self, result=None, error=None):
        self.result = result if result is not None else {"ok": True}
        self.error = error
        self.calls = []

    def execute(self, tool, arguments, *, timeout_seconds):
        self.calls.append((tool.name, arguments, timeout_seconds))
        if self.error is not None:
            raise self.error
        return self.result


def registry():
    result = ToolRegistry()
    result.register(ToolDefinition(
        "echo",
        lambda args: {"value": args.get("value")},
        "echo",
        2.0,
    ))
    return result


def test_task_lifecycle_and_audit():
    audit = AuditTrail()
    coordinator = TaskCoordinator(audit)
    task = AgentTask("t1", "agent-a", "perform a test")
    coordinator.assign(task)
    assert coordinator.status("t1") == "queued"
    coordinator.transition("t1", "running")
    coordinator.transition("t1", "completed")
    assert coordinator.status("t1") == "completed"
    assert audit.verify()


def test_task_lifecycle_rejects_invalid_transition():
    coordinator = TaskCoordinator()
    coordinator.assign(AgentTask("t1", "agent-a", "test"))
    with pytest.raises(ValueError, match="Invalid task transition"):
        coordinator.transition("t1", "completed")


def test_task_lifecycle_rejects_unknown_task():
    with pytest.raises(KeyError):
        TaskCoordinator().transition("missing", "running")


def test_task_lifecycle_rejects_reopening_terminal_task():
    coordinator = TaskCoordinator()
    coordinator.assign(AgentTask("t1", "agent-a", "test"))
    coordinator.transition("t1", "cancelled")
    with pytest.raises(ValueError):
        coordinator.transition("t1", "running")


def test_task_description_must_be_nonempty():
    with pytest.raises(ValueError):
        AgentTask("t1", "agent-a", " ")


def test_tool_definition_rejects_invalid_timeout():
    for value in (0, -1, float("inf"), float("nan"), True):
        with pytest.raises(ValueError):
            ToolDefinition("bad", lambda args: {}, "permission", value)


def test_tool_definition_requires_callable_handler():
    with pytest.raises(TypeError):
        ToolDefinition("bad", None, "permission")


def test_registry_exposes_stable_tool_names():
    tools = registry()
    assert tools.names == ("echo",)


def test_registry_rejects_duplicate_tool():
    tools = registry()
    with pytest.raises(ValueError, match="already registered"):
        tools.register(ToolDefinition("echo", lambda args: {}, "echo"))


def test_agent_state_and_lifecycle():
    agent = AgentRuntime("a", set(), ToolRegistry(), AuditTrail())
    assert agent.state == "stopped"
    agent.start()
    assert agent.state == "active"
    with pytest.raises(RuntimeError, match="already active"):
        agent.start()
    agent.stop()
    assert agent.state == "stopped"
    with pytest.raises(RuntimeError, match="not active"):
        agent.stop()


def test_agent_rejects_invalid_permissions():
    with pytest.raises(ValueError):
        AgentRuntime("a", {""}, ToolRegistry(), AuditTrail())


def test_agent_rejects_invalid_message_recipient():
    agent = AgentRuntime("a", set(), ToolRegistry(), AuditTrail())
    agent.start()
    with pytest.raises(TypeError):
        agent.send("not an agent", "hello")


def test_agent_rejects_blank_message():
    first = AgentRuntime("a", set(), ToolRegistry(), AuditTrail())
    second = AgentRuntime("b", set(), ToolRegistry(), AuditTrail())
    first.start()
    second.start()
    with pytest.raises(ValueError):
        first.send(second, "  ")


def test_agent_rejects_oversized_message():
    first = AgentRuntime("a", set(), ToolRegistry(), AuditTrail())
    second = AgentRuntime("b", set(), ToolRegistry(), AuditTrail())
    first.start()
    second.start()
    with pytest.raises(ValueError, match="maximum"):
        first.send(second, "x" * 1_000_001)


def test_agent_cannot_receive_when_stopped():
    agent = AgentRuntime("a", set(), ToolRegistry(), AuditTrail())
    with pytest.raises(RuntimeError):
        agent.receive()


def test_tool_execution_uses_injected_executor_only():
    executor = RecordingExecutor({"value": "safe"})
    agent = AgentRuntime("a", {"echo"}, registry(), AuditTrail(), executor)
    agent.start()
    result = agent.invoke("echo", {"value": "safe"})
    assert result == {"value": "safe"}
    assert executor.calls[0][0] == "echo"
    assert executor.calls[0][2] == 2.0


def test_tool_execution_rejects_false_isolation_claim():
    executor = RecordingExecutor()
    executor.isolation_verified = False
    agent = AgentRuntime("a", {"echo"}, registry(), AuditTrail(), executor)
    agent.start()
    with pytest.raises(RuntimeError, match="isolation"):
        agent.invoke("echo", {})
    assert executor.calls == []


def test_tool_execution_rejects_isolation_property_error():
    class BrokenExecutor:
        @property
        def isolation_verified(self):
            raise OSError("verification failed")

    agent = AgentRuntime("a", {"echo"}, registry(), AuditTrail(), BrokenExecutor())
    agent.start()
    with pytest.raises(RuntimeError, match="verification failed"):
        agent.invoke("echo", {})
    assert agent.audit.verify()


def test_tool_execution_rejects_non_json_arguments():
    executor = RecordingExecutor()
    agent = AgentRuntime("a", {"echo"}, registry(), AuditTrail(), executor)
    agent.start()
    with pytest.raises(ValueError, match="JSON-compatible"):
        agent.invoke("echo", {"bad": object()})
    assert executor.calls == []


def test_tool_execution_rejects_nonfinite_arguments():
    executor = RecordingExecutor()
    agent = AgentRuntime("a", {"echo"}, registry(), AuditTrail(), executor)
    agent.start()
    with pytest.raises(ValueError):
        agent.invoke("echo", {"bad": math.inf})
    assert executor.calls == []


def test_tool_execution_rejects_oversized_arguments():
    executor = RecordingExecutor()
    agent = AgentRuntime("a", {"echo"}, registry(), AuditTrail(), executor)
    agent.start()
    with pytest.raises(ValueError, match="size limit"):
        agent.invoke("echo", {"data": "x" * 1_000_001})
    assert executor.calls == []


def test_tool_execution_records_backend_failure():
    executor = RecordingExecutor(error=TimeoutError("expired"))
    agent = AgentRuntime("a", {"echo"}, registry(), AuditTrail(), executor)
    agent.start()
    with pytest.raises(TimeoutError):
        agent.invoke("echo", {})
    assert [e["event"] for e in agent.audit.events][-1] == "tool_failed"
    assert agent.audit.verify()


def test_tool_execution_rejects_non_dictionary_result():
    executor = RecordingExecutor(result=["not", "a", "dictionary"])
    agent = AgentRuntime("a", {"echo"}, registry(), AuditTrail(), executor)
    agent.start()
    with pytest.raises(TypeError):
        agent.invoke("echo", {})
    assert agent.audit.verify()


def test_tool_execution_returns_detached_result():
    executor = RecordingExecutor({"nested": {"value": 1}})
    agent = AgentRuntime("a", {"echo"}, registry(), AuditTrail(), executor)
    agent.start()
    result = agent.invoke("echo", {})
    result["nested"]["value"] = 99
    assert executor.result["nested"]["value"] == 1


def test_audit_rejects_non_json_details():
    audit = AuditTrail()
    with pytest.raises(ValueError):
        audit.record("invalid", {"value": object()})


def test_audit_rejects_nonfinite_details():
    audit = AuditTrail()
    with pytest.raises(ValueError):
        audit.record("invalid", {"value": float("nan")})


def test_audit_export_checks_integrity():
    audit = AuditTrail()
    audit.record("event", {"value": 1})
    assert '"event": "event"' in audit.export_json()
    audit._events[0]["event"] = "tampered"
    with pytest.raises(RuntimeError, match="integrity"):
        audit.export_json()


def test_message_validation_rejects_invalid_field_types():
    with pytest.raises(ValueError, match="sender and recipient"):
        AgentMessage(None, "b", "hello", 1.0)

    with pytest.raises(ValueError, match="sender and recipient"):
        AgentMessage("a", 123, "hello", 1.0)

    for timestamp in (True, "1.0", float("nan"), float("inf"), 0):
        with pytest.raises(ValueError, match="timestamp"):
            AgentMessage("a", "b", "hello", timestamp)


def test_send_audit_failure_does_not_deliver_message():
    first = AgentRuntime("a", set(), ToolRegistry(), AuditTrail())
    second = AgentRuntime("b", set(), ToolRegistry(), AuditTrail())
    first.start()
    second.start()

    def fail_record(*args, **kwargs):
        raise OSError("audit unavailable")

    first.audit.record = fail_record

    with pytest.raises(OSError, match="audit unavailable"):
        first.send(second, "hello")

    assert second.receive() == ()


def test_receive_audit_failure_preserves_inbox():
    first = AgentRuntime("a", set(), ToolRegistry(), AuditTrail())
    second = AgentRuntime("b", set(), ToolRegistry(), AuditTrail())
    first.start()
    second.start()
    first.send(second, "hello")

    original_record = second.audit.record

    def fail_record(*args, **kwargs):
        raise OSError("audit unavailable")

    second.audit.record = fail_record

    with pytest.raises(OSError, match="audit unavailable"):
        second.receive()

    second.audit.record = original_record
    messages = second.receive()

    assert len(messages) == 1
    assert messages[0].content == "hello"


def test_task_assignment_audit_failure_does_not_create_task():
    audit = AuditTrail()
    coordinator = TaskCoordinator(audit)

    def fail_record(*args, **kwargs):
        raise OSError("audit unavailable")

    audit.record = fail_record

    task = AgentTask("t1", "agent-a", "test")
    with pytest.raises(OSError, match="audit unavailable"):
        coordinator.assign(task)

    assert coordinator.tasks == ()
    with pytest.raises(KeyError):
        coordinator.get("t1")


def test_task_transition_audit_failure_preserves_status():
    audit = AuditTrail()
    coordinator = TaskCoordinator(audit)
    coordinator.assign(AgentTask("t1", "agent-a", "test"))

    original_record = audit.record

    def fail_record(*args, **kwargs):
        raise OSError("audit unavailable")

    audit.record = fail_record

    with pytest.raises(OSError, match="audit unavailable"):
        coordinator.transition("t1", "running")

    assert coordinator.status("t1") == "queued"

    audit.record = original_record
    coordinator.transition("t1", "running")
    assert coordinator.status("t1") == "running"
