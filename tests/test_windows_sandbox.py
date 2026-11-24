import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import aviy.agents.windows_sandbox as windows_sandbox_module
from aviy.agents.runtime import SandboxToolDefinition, ToolDefinition, ToolRegistry
from aviy.agents.windows_sandbox import (
    WindowsSandboxError,
    WindowsSandboxExecutor,
)


def valid_envelope(result=None):
    return {
        "status": "executed",
        "nonce": "expected-nonce",
        "computer": "SANDBOX-GUEST",
        "identity": r"SANDBOX-GUEST\WDAGUtilityAccount",
        "result": result if result is not None else {"ok": True},
    }


def test_sandbox_tool_definition_accepts_script():
    tool = SandboxToolDefinition(
        "echo",
        "param($Arguments)\n[ordered]@{ value = $Arguments.value }",
        "echo",
    )
    assert tool.name == "echo"
    assert "Arguments" in tool.script


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "script": "x", "required_permission": "p"},
        {"name": "x", "script": " ", "required_permission": "p"},
        {"name": "x", "script": "x", "required_permission": ""},
        {"name": "x", "script": "x", "required_permission": "p", "timeout_seconds": 0},
        {
            "name": "x",
            "script": "x",
            "required_permission": "p",
            "timeout_seconds": float("inf"),
        },
        {
            "name": "x",
            "script": "x",
            "required_permission": "p",
            "timeout_seconds": True,
        },
    ],
)
def test_sandbox_tool_definition_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SandboxToolDefinition(**kwargs)


def test_sandbox_tool_definition_rejects_oversized_script():
    with pytest.raises(ValueError, match="size limit"):
        SandboxToolDefinition("large", "x" * 1_000_001, "permission")


def test_registry_accepts_both_tool_types():
    registry = ToolRegistry()
    registry.register(ToolDefinition("python", lambda args: {}, "python"))
    registry.register(
        SandboxToolDefinition("powershell", "param($Arguments) @{}", "ps")
    )
    assert registry.names == ("powershell", "python")


def test_registry_rejects_duplicate_names_across_tool_types():
    registry = ToolRegistry()
    registry.register(ToolDefinition("same", lambda args: {}, "permission"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            SandboxToolDefinition("same", "param($Arguments) @{}", "permission")
        )


def test_windows_executor_rejects_python_callable_tool():
    executor = WindowsSandboxExecutor(executable="unused")
    tool = ToolDefinition("python", lambda args: {}, "permission")
    with pytest.raises(TypeError, match="SandboxToolDefinition"):
        executor.execute(tool, {}, timeout_seconds=1)


def test_configuration_disables_network_clipboard_and_vgpu(tmp_path):
    input_folder = tmp_path / "Input"
    output_folder = tmp_path / "Output"
    input_folder.mkdir()
    output_folder.mkdir()

    configuration = WindowsSandboxExecutor._configuration(input_folder, output_folder)
    root = ET.fromstring(configuration)

    assert root.findtext("Networking") == "Disable"
    assert root.findtext("ClipboardRedirection") == "Disable"
    assert root.findtext("vGPU") == "Disable"

    mappings = root.findall("./MappedFolders/MappedFolder")
    assert len(mappings) == 2
    assert mappings[0].findtext("ReadOnly") == "true"
    assert mappings[1].findtext("ReadOnly") == "false"
    assert mappings[0].findtext("SandboxFolder") == r"C:\AviyInput"
    assert mappings[1].findtext("SandboxFolder") == r"C:\AviyOutput"


def test_guest_result_accepts_verified_guest_and_json_result():
    result = WindowsSandboxExecutor._validate_guest_result(
        valid_envelope({"answer": 42}),
        nonce="expected-nonce",
        host_computer="HOST-PC",
    )
    assert result == {"answer": 42}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda item: item.update(status="error"), "successful"),
        (lambda item: item.update(nonce="wrong"), "nonce"),
        (lambda item: item.update(computer="HOST-PC"), "computer"),
        (lambda item: item.update(identity=r"HOST-PC\Administrator"), "account"),
        (lambda item: item.update(result=["not", "an", "object"]), "JSON object"),
    ],
)
def test_guest_result_rejects_invalid_envelopes(mutate, message):
    envelope = valid_envelope()
    mutate(envelope)
    with pytest.raises(WindowsSandboxError, match=message):
        WindowsSandboxExecutor._validate_guest_result(
            envelope,
            nonce="expected-nonce",
            host_computer="HOST-PC",
        )


def test_guest_result_rejects_non_json_values():
    envelope = valid_envelope({"bad": float("nan")})
    with pytest.raises(WindowsSandboxError, match="finite JSON"):
        WindowsSandboxExecutor._validate_guest_result(
            envelope,
            nonce="expected-nonce",
            host_computer="HOST-PC",
        )


def test_guest_result_rejects_oversized_result():
    envelope = valid_envelope({"data": "x" * 1_000_001})
    with pytest.raises(WindowsSandboxError, match="size limit"):
        WindowsSandboxExecutor._validate_guest_result(
            envelope,
            nonce="expected-nonce",
            host_computer="HOST-PC",
        )


def test_guest_result_rejects_non_object_envelope():
    with pytest.raises(WindowsSandboxError, match="JSON object"):
        WindowsSandboxExecutor._validate_guest_result(
            ["invalid"],
            nonce="expected-nonce",
            host_computer="HOST-PC",
        )


def test_execute_removes_temporary_directory_after_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        WindowsSandboxExecutor,
        "isolation_verified",
        property(lambda self: True),
    )

    class CompletedProcess:
        pid = 12345
        returncode = 0

        @staticmethod
        def poll():
            return 0

    def fake_popen(arguments, **kwargs):
        config_path = Path(arguments[1])
        root = config_path.parent
        request = json.loads(
            (root / "Input" / "request.json").read_text(encoding="utf-8")
        )
        envelope = valid_envelope()
        envelope["nonce"] = request["nonce"]
        (root / "Output" / "result.json").write_text(
            json.dumps(envelope), encoding="utf-8"
        )
        return CompletedProcess()

    monkeypatch.setattr(windows_sandbox_module.subprocess, "Popen", fake_popen)

    executor = WindowsSandboxExecutor(
        executable=tmp_path / "WindowsSandbox.exe",
        temporary_directory=tmp_path,
    )
    tool = SandboxToolDefinition(
        "echo",
        "param($Arguments)\n[ordered]@{ ok = $true }",
        "permission",
    )

    result = executor.execute(tool, {}, timeout_seconds=1)

    assert result == {"ok": True}
    assert list(tmp_path.glob("AviySandbox_*")) == []


def test_execute_removes_temporary_directory_when_launcher_fails(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        WindowsSandboxExecutor,
        "isolation_verified",
        property(lambda self: True),
    )

    class FailedProcess:
        pid = 12346
        returncode = 7

        @staticmethod
        def poll():
            return 7

    monkeypatch.setattr(
        windows_sandbox_module.subprocess,
        "Popen",
        lambda *args, **kwargs: FailedProcess(),
    )

    executor = WindowsSandboxExecutor(
        executable=tmp_path / "WindowsSandbox.exe",
        temporary_directory=tmp_path,
    )
    tool = SandboxToolDefinition("echo", "param($Arguments) @{}", "permission")

    with pytest.raises(WindowsSandboxError, match="exited with code 7"):
        executor.execute(tool, {}, timeout_seconds=1)

    assert list(tmp_path.glob("AviySandbox_*")) == []


def test_execute_removes_temporary_directory_when_launcher_cannot_start(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        WindowsSandboxExecutor,
        "isolation_verified",
        property(lambda self: True),
    )

    def failed_popen(*args, **kwargs):
        raise OSError("simulated launcher failure")

    monkeypatch.setattr(
        windows_sandbox_module.subprocess,
        "Popen",
        failed_popen,
    )

    executor = WindowsSandboxExecutor(
        executable=tmp_path / "WindowsSandbox.exe",
        temporary_directory=tmp_path,
    )
    tool = SandboxToolDefinition("echo", "param($Arguments) @{}", "permission")

    with pytest.raises(WindowsSandboxError, match="Could not start"):
        executor.execute(tool, {}, timeout_seconds=1)

    assert list(tmp_path.glob("AviySandbox_*")) == []
