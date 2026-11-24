from __future__ import annotations

import json
import math
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import xml.sax.saxutils
from pathlib import Path

from aviy.agents.runtime import (
    MAX_TOOL_ARGUMENT_BYTES,
    MAX_TOOL_RESULT_BYTES,
    SandboxToolDefinition,
)

_RUNNER_SCRIPT = r"""
$ErrorActionPreference = "Stop"
$inputRoot = "C:\AviyInput"
$outputRoot = "C:\AviyOutput"
$resultPath = Join-Path $outputRoot "result.json"
$exitCode = 0
$requestNonce = ""

try {
    $request = Get-Content -LiteralPath (Join-Path $inputRoot "request.json") -Raw |
        ConvertFrom-Json
    $requestNonce = $request.nonce
    $arguments = $request.arguments
    $toolPath = Join-Path $inputRoot "tool.ps1"

    $toolOutput = @(& $toolPath -Arguments $arguments)

    if ($toolOutput.Count -ne 1) {
        throw "A sandbox tool must return exactly one result."
    }

    if ($toolOutput[0] -isnot [System.Collections.IDictionary]) {
        throw "A sandbox tool must return a PowerShell hashtable."
    }

    $envelope = [ordered]@{
        status = "executed"
        nonce = $request.nonce
        computer = $env:COMPUTERNAME
        identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
        timestamp_utc = [DateTime]::UtcNow.ToString("o")
        result = $toolOutput[0]
    }
}
catch {
    $exitCode = 1
    $envelope = [ordered]@{
        status = "error"
        nonce = $requestNonce
        computer = $env:COMPUTERNAME
        identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
        error_type = $_.Exception.GetType().FullName
    }
}

try {
    $json = ConvertTo-Json -InputObject $envelope -Depth 32 -Compress
    Set-Content -LiteralPath $resultPath -Value $json -Encoding UTF8
}
catch {
    $exitCode = 1
}

Start-Sleep -Seconds 2
shutdown.exe /s /t 0 /f
exit $exitCode
"""


class WindowsSandboxError(RuntimeError):
    """Windows Sandbox execution or verification failed."""


class WindowsSandboxExecutor:
    """Execute approved PowerShell tools in disposable Windows Sandbox guests.

    The read-only input mapping contains only the tool script, JSON arguments,
    and this runner. The writable output mapping contains only the guest result.
    A successful result is accepted only after checking the guest identity,
    guest computer name, per-run nonce, launcher exit code, and JSON structure.

    The nonce is a run-correlation value, not cryptographic attestation.
    """

    STARTUP_GRACE_SECONDS = 90
    RESULT_WAIT_SECONDS = 15

    def __init__(
        self,
        *,
        executable: str | Path | None = None,
        temporary_directory: str | Path | None = None,
    ) -> None:
        default_executable = (
            Path(os.environ.get("WINDIR", r"C:\Windows"))
            / "System32"
            / "WindowsSandbox.exe"
        )
        self.executable = Path(executable) if executable else default_executable
        self.temporary_directory = (
            Path(temporary_directory) if temporary_directory else None
        )

    @property
    def isolation_verified(self) -> bool:
        """Report host capability; each execution separately verifies its guest."""
        return sys.platform == "win32" and self.executable.is_file()

    @staticmethod
    def _configuration(input_folder: Path, output_folder: Path) -> str:
        host_input = xml.sax.saxutils.escape(str(input_folder.resolve()))
        host_output = xml.sax.saxutils.escape(str(output_folder.resolve()))

        return f"""<Configuration>
    <Networking>Disable</Networking>
    <ClipboardRedirection>Disable</ClipboardRedirection>
    <vGPU>Disable</vGPU>
    <MappedFolders>
        <MappedFolder>
            <HostFolder>{host_input}</HostFolder>
            <SandboxFolder>C:\\AviyInput</SandboxFolder>
            <ReadOnly>true</ReadOnly>
        </MappedFolder>
        <MappedFolder>
            <HostFolder>{host_output}</HostFolder>
            <SandboxFolder>C:\\AviyOutput</SandboxFolder>
            <ReadOnly>false</ReadOnly>
        </MappedFolder>
    </MappedFolders>
    <LogonCommand>
        <Command>powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File C:\\AviyInput\\runner.ps1</Command>
    </LogonCommand>
</Configuration>
"""

    @staticmethod
    def _validate_guest_result(
        envelope: object,
        *,
        nonce: str,
        host_computer: str,
    ) -> dict[str, object]:
        if not isinstance(envelope, dict):
            raise WindowsSandboxError("Sandbox result must be a JSON object.")

        if envelope.get("status") != "executed":
            raise WindowsSandboxError(
                "Sandbox guest did not report successful tool execution."
            )

        if envelope.get("nonce") != nonce:
            raise WindowsSandboxError("Sandbox result nonce did not match.")

        guest_computer = envelope.get("computer")
        if (
            not isinstance(guest_computer, str)
            or not guest_computer.strip()
            or guest_computer.casefold() == host_computer.casefold()
        ):
            raise WindowsSandboxError(
                "Sandbox guest computer identity could not be verified."
            )

        identity = envelope.get("identity")
        if not isinstance(identity, str) or not identity.casefold().endswith(
            r"\wdagutilityaccount"
        ):
            raise WindowsSandboxError(
                "Sandbox guest account identity could not be verified."
            )

        result = envelope.get("result")
        if not isinstance(result, dict):
            raise WindowsSandboxError("Sandbox tool must return a JSON object.")

        try:
            encoded = json.dumps(result, allow_nan=False, separators=(",", ":")).encode(
                "utf-8"
            )
        except (TypeError, ValueError) as exc:
            raise WindowsSandboxError(
                "Sandbox result is not finite JSON data."
            ) from exc

        if len(encoded) > MAX_TOOL_RESULT_BYTES:
            raise WindowsSandboxError("Sandbox result exceeds the size limit.")

        return json.loads(encoded.decode("utf-8"))

    def execute(
        self,
        tool: SandboxToolDefinition,
        arguments: dict[str, object],
        *,
        timeout_seconds: float,
    ) -> dict[str, object]:
        if not isinstance(tool, SandboxToolDefinition):
            raise TypeError(
                "Windows Sandbox accepts only SandboxToolDefinition instances."
            )
        if not self.isolation_verified:
            raise WindowsSandboxError("Windows Sandbox is unavailable on this host.")
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a dictionary.")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Execution timeout must be finite and positive.")

        try:
            encoded_arguments = json.dumps(
                arguments,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("Tool arguments must be finite JSON data.") from exc

        if len(encoded_arguments) > MAX_TOOL_ARGUMENT_BYTES:
            raise ValueError("Tool arguments exceed the size limit.")

        nonce = secrets.token_hex(32)

        temporary_root = tempfile.mkdtemp(
            prefix="AviySandbox_",
            dir=self.temporary_directory,
        )
        root = Path(temporary_root)
        try:
            input_folder = root / "Input"
            output_folder = root / "Output"
            input_folder.mkdir()
            output_folder.mkdir()

            (input_folder / "tool.ps1").write_text(tool.script, encoding="utf-8")
            (input_folder / "runner.ps1").write_text(_RUNNER_SCRIPT, encoding="utf-8")
            request = {"nonce": nonce, "arguments": arguments}
            (input_folder / "request.json").write_text(
                json.dumps(request, allow_nan=False),
                encoding="utf-8",
            )

            configuration = self._configuration(input_folder, output_folder)
            wsb_path = root / "AviyTool.wsb"
            wsb_path.write_text(configuration, encoding="utf-8")

            try:
                process = subprocess.Popen(
                    [str(self.executable), str(wsb_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                raise WindowsSandboxError("Could not start Windows Sandbox.") from exc

            result_path = output_folder / "result.json"
            deadline = (
                time.monotonic() + float(timeout_seconds) + self.STARTUP_GRACE_SECONDS
            )
            previous_size = None
            stable_polls = 0

            # The Sandbox launcher may exit before its guest has finished writing
            # the mapped output. Wait for both a stable result and launcher exit.
            while time.monotonic() < deadline:
                if result_path.is_file() and not result_path.is_symlink():
                    current_size = result_path.stat().st_size
                    if current_size > 0 and current_size == previous_size:
                        stable_polls += 1
                    else:
                        stable_polls = 0
                    previous_size = current_size
                else:
                    previous_size = None
                    stable_polls = 0

                if stable_polls >= 3 and process.poll() is not None:
                    break

                time.sleep(0.1)

            if process.poll() is None:
                try:
                    subprocess.run(
                        [
                            "taskkill.exe",
                            "/PID",
                            str(process.pid),
                            "/T",
                            "/F",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
                raise TimeoutError(
                    "Windows Sandbox did not finish within the allowed time."
                )

            if process.returncode != 0:
                raise WindowsSandboxError(
                    f"Windows Sandbox launcher exited with code {process.returncode}."
                )

            if not result_path.is_file():
                raise WindowsSandboxError(
                    f"Sandbox guest did not produce a result file. "
                    f"Diagnostic files preserved at: {root}"
                )

            if result_path.is_symlink():
                raise WindowsSandboxError("Sandbox result cannot be a symbolic link.")
            if result_path.stat().st_size > MAX_TOOL_RESULT_BYTES + 16_384:
                raise WindowsSandboxError("Sandbox result file exceeds the size limit.")

            try:
                envelope = json.loads(result_path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise WindowsSandboxError("Sandbox result file is invalid.") from exc

            result = self._validate_guest_result(
                envelope,
                nonce=nonce,
                host_computer=socket.gethostname(),
            )
            return result
        finally:
            shutil.rmtree(root, ignore_errors=True)
