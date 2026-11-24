import hashlib
from typing import Any, cast

import pytest

from aviy.agents.runtime import (
    AgentRuntime,
    AgentTask,
    AuditTrail,
    TaskCoordinator,
    ToolDefinition,
    ToolRegistry,
)
from aviy.autonomous.self_improvement import (
    CandidateAdoptionLedger,
    CandidateExperimentRunner,
    ImprovementObjective,
)
from aviy.infrastructure.updates import (
    UpdateFile,
    UpdateManager,
    UpdateManifest,
    canonical_manifest_payload,
)
from aviy.knowledge.research import (
    Citation,
    ResearchEngine,
    ResearchSource,
    ResourceAcquirer,
)
from aviy.models.specialists import (
    BenchmarkCase,
    MathematicsSpecialist,
    ProgrammingSpecialist,
    ScienceSpecialist,
    SpecialistBenchmark,
    SpecialistResult,
    SpecialistRouter,
)

# ============================================================
# Phase 16: Specialist models
# ============================================================


def test_mathematics_specialist_evaluates_safe_arithmetic():
    specialist = MathematicsSpecialist()
    assert specialist.evaluate_expression("2 + 3 * 4") == 14
    assert specialist.evaluate_expression("(10 - 4) / 2") == 3


def test_mathematics_specialist_accepts_calculate_instruction():
    result = MathematicsSpecialist().analyze("calculate 4 + 5")
    assert result.answer == "9.0"
    assert result.confidence == 1.0


@pytest.mark.parametrize(
    ("equation", "expected"),
    [
        ("2*x + 4 = 10", 3),
        ("3*x - 6 = 9", 5),
        ("x / 2 = 5", 10),
        ("2*(x + 3) = 14", 4),
        ("-2*x = 8", -4),
        ("x + 5 = 5", 0),
    ],
)
def test_mathematics_specialist_solves_linear_equations(equation, expected):
    result = MathematicsSpecialist().solve_linear_equation(equation)
    assert result["x"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "equation",
    [
        "x*x = 4",
        "x + 1 = x + 1",
        "x + 1 = x + 2",
        "x / 0 = 2",
        "__import__('os').system('whoami') = 1",
        "2**100 = 4",
        "x = 1; import os",
        "x + 1",
    ],
)
def test_mathematics_specialist_rejects_unsafe_or_invalid_equations(equation):
    with pytest.raises((ValueError, SyntaxError, ArithmeticError)):
        MathematicsSpecialist().solve_linear_equation(equation)


def test_mathematics_specialist_rejects_unsafe_expression():
    specialist = MathematicsSpecialist()
    with pytest.raises(ValueError):
        specialist.evaluate_expression("__import__('os').system('whoami')")


def test_programming_specialist_uses_static_analysis_only():
    result = ProgrammingSpecialist().analyze(
        "def greet(name):\n    return 'Hello ' + name\n"
    )
    assert result.metadata is not None
    assert result.metadata["valid_syntax"] is True
    assert result.metadata["executed"] is False
    assert result.metadata["functions"] == 1


def test_programming_specialist_flags_dangerous_calls():
    result = ProgrammingSpecialist().analyze("value = eval('1 + 1')\n")
    assert result.metadata is not None
    assert "eval" in cast(list[str], result.metadata["review_required_calls"])
    assert result.metadata["executed"] is False


def test_science_specialist_does_not_claim_unverified_facts():
    result = ScienceSpecialist().analyze("Does this experiment prove the hypothesis?")
    assert result.evidence_required is True
    assert result.metadata is not None
    assert result.metadata["validated_claim"] is False
    assert result.confidence == 0.0


def test_specialist_router_routes_and_collaborates():
    specialists = [
        MathematicsSpecialist(),
        ProgrammingSpecialist(),
        ScienceSpecialist(),
    ]
    router = SpecialistRouter(specialists)

    assert "mathematics" in router.domains
    routed = router.route("calculate 4 + 5")
    assert len(routed) == 1
    assert routed[0].domain == "mathematics"

    results = router.collaborate("calculate 4 + 5")
    assert len(results) == 1
    assert results[0].specialist == "mathematics"
    assert results[0].answer == "9.0"


def test_specialist_benchmark_reports_accuracy():
    specialist = MathematicsSpecialist()
    cases = [
        BenchmarkCase("2 + 2", "4.0"),
        BenchmarkCase("3 * 3", "9.0"),
    ]
    report = SpecialistBenchmark().evaluate(specialist, cases, minimum_accuracy=1.0)
    assert report.passed == 2
    assert report.total == 2
    assert report.accepted is True


def test_specialist_benchmark_rejects_false_positive_substring():
    specialist = MathematicsSpecialist()
    report = SpecialistBenchmark().evaluate(
        specialist,
        [BenchmarkCase("2 + 2", expected_exact="5.0")],
        minimum_accuracy=1.0,
    )
    assert report.accepted is False
    assert report.case_results[0].passed is False
    assert "Exact answer mismatch" in report.case_results[0].reason


def test_specialist_benchmark_checks_numeric_tolerance():
    specialist = MathematicsSpecialist()
    report = SpecialistBenchmark().evaluate(
        specialist,
        [
            BenchmarkCase(
                "2 + 2",
                expected_numeric=4.0,
                absolute_tolerance=1e-8,
            )
        ],
        minimum_accuracy=1.0,
    )
    assert report.accepted is True
    assert report.case_results[0].reason == "All criteria passed."


def test_specialist_benchmark_checks_required_metadata():
    specialist = ProgrammingSpecialist()
    report = SpecialistBenchmark().evaluate(
        specialist,
        [
            BenchmarkCase(
                "def greet():\n    return 'hello'",
                expected_exact=(
                    "Python syntax is valid; static inspection completed "
                    "without execution."
                ),
                required_metadata={
                    "valid_syntax": True,
                    "executed": False,
                    "analysis_scope": "static_python_ast",
                },
            )
        ],
        minimum_accuracy=1.0,
    )
    assert report.accepted is True


def test_specialist_benchmark_reports_metadata_failures():
    specialist = ScienceSpecialist()
    report = SpecialistBenchmark().evaluate(
        specialist,
        [
            BenchmarkCase(
                "Does this prove the hypothesis?",
                required_metadata={
                    "validated_claim": True,
                    "evidence_status": "verified",
                },
            )
        ],
        minimum_accuracy=1.0,
    )
    assert report.accepted is False
    assert report.case_results[0].passed is False
    assert "Metadata mismatch: validated_claim" in report.case_results[0].reason


@pytest.mark.parametrize(
    "case",
    [
        BenchmarkCase("2 + 2"),
        BenchmarkCase("", expected_exact="4.0"),
        BenchmarkCase("2 + 2", expected_numeric=float("nan")),
        BenchmarkCase("2 + 2", expected_numeric=4, absolute_tolerance=-1),
    ],
)
def test_specialist_benchmark_rejects_invalid_cases(case):
    with pytest.raises(ValueError):
        SpecialistBenchmark().evaluate(
            MathematicsSpecialist(), [case], minimum_accuracy=1.0
        )


def test_specialist_result_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        SpecialistResult("test", "answer", 1.5)


# ============================================================
# Phase 17: Agents, tools and audit
# ============================================================


def make_tool_registry():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            handler=lambda arguments: {"value": arguments["value"]},
            required_permission="echo",
        )
    )
    return registry


def test_agent_lifecycle_and_message_exchange():
    audit_a = AuditTrail()
    audit_b = AuditTrail()
    tools = ToolRegistry()

    first = AgentRuntime("agent-a", set(), tools, audit_a)
    second = AgentRuntime("agent-b", set(), tools, audit_b)

    first.start()
    second.start()
    first.send(second, "hello")
    messages = second.receive()

    assert len(messages) == 1
    assert messages[0].sender == "agent-a"
    assert messages[0].content == "hello"
    assert audit_a.verify()
    assert audit_b.verify()


def test_agent_cannot_invoke_tool_without_isolation():
    runtime = AgentRuntime("agent-a", {"echo"}, make_tool_registry(), AuditTrail())
    runtime.start()

    with pytest.raises(RuntimeError, match="isolation"):
        runtime.invoke("echo", {"value": "hello"})


def test_agent_permission_is_checked_before_tool_execution():
    runtime = AgentRuntime("agent-a", set(), make_tool_registry(), AuditTrail())
    runtime.start()

    with pytest.raises(PermissionError):
        runtime.invoke("echo", {"value": "hello"})


def test_audit_trail_detects_internal_tampering():
    audit = AuditTrail()
    audit.record("test", {"nested": {"value": 1}})
    assert audit.verify()

    cast(Any, audit._events)[0]["details"]["nested"]["value"] = 2
    assert audit.verify() is False


def test_audit_events_cannot_mutate_internal_nested_details():
    audit = AuditTrail()
    audit.record("test", {"nested": {"value": 1}})

    returned = audit.events
    cast(Any, returned)[0]["details"]["nested"]["value"] = 99

    assert cast(Any, audit.events)[0]["details"]["nested"]["value"] == 1
    assert audit.verify()


def test_task_coordinator_rejects_duplicate_task_ids():
    coordinator = TaskCoordinator()
    task = AgentTask("task-1", "agent-a", "Test task")
    coordinator.assign(task)

    with pytest.raises(ValueError):
        coordinator.assign(task)

    assert coordinator.get("task-1") == task


# ============================================================
# Phase 18: Research and knowledge acquisition
# ============================================================


def make_acquirer(maximum_bytes=1000):
    return ResourceAcquirer(
        allowed_licenses={"MIT"},
        allowed_hosts={"example.com"},
        maximum_bytes=maximum_bytes,
    )


def test_resource_acquirer_preserves_hash_and_citation():
    content = b"Research content"
    citation = Citation(
        "cite-1",
        "https://example.com/paper",
        "Research paper",
        "Example Publisher",
        "page 2",
    )
    source = make_acquirer().acquire(
        source_id="source-1",
        uri="https://example.com/data",
        title="Research data",
        license="MIT",
        content=content,
        publisher="Example Publisher",
        citations=(citation,),
    )

    assert source.content == content.decode("utf-8")
    assert source.content_sha256 == hashlib.sha256(content).hexdigest()
    assert source.verified is False
    assert len(source.citations) == 1
    assert source.citations[0].citation_id == "cite-1"


def test_resource_acquirer_rejects_unapproved_host():
    with pytest.raises(PermissionError):
        make_acquirer().acquire(
            source_id="s1",
            uri="https://unapproved.example/data",
            title="Data",
            license="MIT",
            content=b"data",
        )


def test_resource_acquirer_rejects_unapproved_license():
    with pytest.raises(PermissionError):
        make_acquirer().acquire(
            source_id="s1",
            uri="https://example.com/data",
            title="Data",
            license="UNAPPROVED",
            content=b"data",
        )


def test_resource_acquirer_rejects_oversized_content():
    with pytest.raises(ValueError):
        make_acquirer(maximum_bytes=4).acquire(
            source_id="s1",
            uri="https://example.com/data",
            title="Data",
            license="MIT",
            content=b"too much content",
        )


def test_resource_acquirer_rejects_unsafe_uri():
    with pytest.raises(ValueError):
        make_acquirer().acquire(
            source_id="s1",
            uri="file:///private/data",
            title="Data",
            license="MIT",
            content=b"data",
        )


def test_research_engine_preserves_verification_and_citations():
    content = "A research finding."
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    citation = Citation("cite-1", "https://example.com/paper", "Paper")
    source = ResearchSource(
        source_id="source-1",
        uri="https://example.com/data",
        title="Research data",
        license="MIT",
        content=content,
        content_sha256=digest,
        verified=False,
        uncertainty="Not yet verified",
        publisher="Example Publisher",
        retrieved_at="2026-01-01T00:00:00+00:00",
        citations=(citation,),
    )

    engine = ResearchEngine(
        discover=lambda query: [source],
        verify=lambda item: True,
        allowed_licenses={"MIT"},
    )
    results = engine.run("research question")

    assert len(results) == 1
    assert results[0].verified is True
    assert len(results[0].citations) == 1
    assert "corroboration" in results[0].uncertainty
    assert "Not yet verified" in results[0].uncertainty


def test_research_engine_rejects_content_hash_mismatch():
    source = ResearchSource(
        "source-1",
        "https://example.com/data",
        "Data",
        "MIT",
        "changed content",
        "0" * 64,
        False,
        "Unverified",
    )
    engine = ResearchEngine(
        discover=lambda query: [source],
        verify=lambda item: True,
        allowed_licenses={"MIT"},
    )

    with pytest.raises(ValueError, match="hash"):
        engine.run("research question")


# ============================================================
# Phase 19: Candidate self-improvement
# ============================================================


class TestOnlyExperimentBackend:
    """A test double; this does not provide real OS isolation."""

    isolation_verified = True

    def run(self, experiment, workspace, *, role, repeat):
        return experiment(workspace)


def make_objective():
    return ImprovementObjective(
        name="accuracy",
        metric="accuracy",
        minimum_gain=0.05,
        maximum_regression=0.0,
        direction="maximize",
        reproducibility_tolerance=0.001,
    )


def test_candidate_runner_fails_closed_without_backend(tmp_path):
    runner = CandidateExperimentRunner(tmp_path)

    with pytest.raises(RuntimeError, match="isolation"):
        runner.evaluate(
            "candidate-1",
            (make_objective(),),
            lambda workspace: {"accuracy": 0.5},
            lambda workspace: {"accuracy": 0.8},
            repeats=2,
        )


def test_candidate_runner_evaluates_repeated_candidate(tmp_path):
    runner = CandidateExperimentRunner(tmp_path, TestOnlyExperimentBackend())
    result = runner.evaluate(
        "candidate-1",
        (make_objective(),),
        lambda workspace: {"accuracy": 0.5},
        lambda workspace: {"accuracy": 0.8},
        repeats=3,
        configuration={"version": 1},
    )

    assert result.reproducible is True
    assert result.adopted_eligible is True
    assert result.approval_required is True
    assert result.metric_gains is not None
    assert result.metric_gains["accuracy"] == pytest.approx(0.3)
    assert len(result.baseline_scores["accuracy"]) == 3
    assert len(result.candidate_scores["accuracy"]) == 3


def test_candidate_runner_rejects_insufficient_improvement(tmp_path):
    runner = CandidateExperimentRunner(tmp_path, TestOnlyExperimentBackend())
    result = runner.evaluate(
        "candidate-2",
        (make_objective(),),
        lambda workspace: {"accuracy": 0.5},
        lambda workspace: {"accuracy": 0.51},
        repeats=2,
    )

    assert result.adopted_eligible is False
    assert any("Insufficient improvement" in reason for reason in result.reasons)


def test_candidate_runner_rejects_non_reproducible_results(tmp_path):
    runner = CandidateExperimentRunner(tmp_path, TestOnlyExperimentBackend())

    def variable_candidate(workspace):
        index = int(workspace.name.split("-")[-1])
        return {"accuracy": 0.7 + index * 0.1}

    result = runner.evaluate(
        "candidate-3",
        (make_objective(),),
        lambda workspace: {"accuracy": 0.5},
        variable_candidate,
        repeats=3,
    )

    assert result.reproducible is False
    assert result.adopted_eligible is False


# ============================================================
# Phase 20: Secure update management
# ============================================================


def make_manifest(relative_path="component/module.txt", content=b"new content"):
    update_file = UpdateFile(
        relative_path=relative_path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
    )
    unsigned = UpdateManifest(
        update_id="update-1",
        component="core",
        version="1.0.1",
        files=(update_file,),
        signature=b"test-signature",
        signed_payload=b"",
    )
    return UpdateManifest(
        update_id=unsigned.update_id,
        component=unsigned.component,
        version=unsigned.version,
        files=unsigned.files,
        signature=unsigned.signature,
        signed_payload=canonical_manifest_payload(unsigned),
    )


def make_update_manager(root, verifier=None):
    if verifier is None:
        verifier = lambda payload, signature: signature == b"test-signature"
    return UpdateManager(
        root,
        trusted_signature_verifier=verifier,
        allowed_components={"core"},
    )


def test_update_manager_verifies_manifest():
    manager = make_update_manager("unused-update-test")
    manifest = make_manifest()
    manager.verify(manifest)


def test_update_manager_rejects_bad_signature(tmp_path):
    manager = make_update_manager(
        tmp_path,
        verifier=lambda payload, signature: False,
    )

    with pytest.raises(PermissionError, match="signature"):
        manager.verify(make_manifest())


def test_update_manager_rejects_tampered_signed_payload(tmp_path):
    manager = make_update_manager(tmp_path)
    manifest = make_manifest()

    tampered = UpdateManifest(
        update_id=manifest.update_id,
        component=manifest.component,
        version=manifest.version,
        files=(
            UpdateFile(
                "component/module.txt",
                b"tampered",
                hashlib.sha256(b"tampered").hexdigest(),
            ),
        ),
        signature=manifest.signature,
        signed_payload=manifest.signed_payload,
    )

    with pytest.raises(PermissionError, match="payload"):
        manager.verify(tampered)


def test_update_manager_rejects_path_traversal(tmp_path):
    manager = make_update_manager(tmp_path)
    manifest = make_manifest("../outside.txt")

    with pytest.raises(ValueError, match="Unsafe"):
        manager.verify(manifest)


def test_update_manager_requires_explicit_approval(tmp_path):
    manager = make_update_manager(tmp_path)

    with pytest.raises(PermissionError, match="approval"):
        manager.apply(
            make_manifest(),
            compatibility_check=lambda manifest: True,
        )


def test_update_manager_applies_approved_update(tmp_path):
    manager = make_update_manager(tmp_path)
    result = manager.apply(
        make_manifest(),
        compatibility_check=lambda manifest: True,
        approval=lambda manifest: True,
    )

    destination = tmp_path / "component" / "module.txt"
    assert result.applied == ("component/module.txt",)
    assert destination.read_bytes() == b"new content"


def test_update_manager_rejects_unapproved_component(tmp_path):
    manager = make_update_manager(tmp_path)
    manifest = make_manifest()

    unapproved = UpdateManifest(
        update_id=manifest.update_id,
        component="unapproved",
        version=manifest.version,
        files=manifest.files,
        signature=manifest.signature,
        signed_payload=manifest.signed_payload,
    )

    with pytest.raises(PermissionError, match="Component"):
        manager.verify(unapproved)


def test_mathematics_specialist_meets_initial_domain_benchmark():
    from aviy.models.specialists import (
        BenchmarkCase,
        MathematicsSpecialist,
        SpecialistBenchmark,
    )

    cases = [
        BenchmarkCase("calculate 2 + 2", expected_numeric=4),
        BenchmarkCase("evaluate (10 - 4) / 2", expected_numeric=3),
        BenchmarkCase("solve 2*x + 4 = 10", expected_numeric=3),
        BenchmarkCase("solve 3*x - 6 = 9", expected_numeric=5),
        BenchmarkCase("solve x / 2 = 5", expected_numeric=10),
        BenchmarkCase("solve 2*(x + 3) = 14", expected_numeric=4),
        BenchmarkCase("solve -2*x = 8", expected_numeric=-4),
        BenchmarkCase("solve x + 5 = 5", expected_numeric=0),
        BenchmarkCase("calculate 0.1 + 0.2", expected_numeric=0.3),
    ]

    report = SpecialistBenchmark().evaluate(
        MathematicsSpecialist(), cases, minimum_accuracy=1.0
    )
    assert report.domain == "mathematics"
    assert report.total == 9
    assert report.accepted is True, report.case_results


def test_programming_specialist_meets_initial_domain_benchmark():
    from aviy.models.specialists import (
        BenchmarkCase,
        ProgrammingSpecialist,
        SpecialistBenchmark,
    )

    cases = [
        BenchmarkCase(
            "def greet(name):\n    return name",
            required_metadata={
                "valid_syntax": True,
                "functions": 1,
                "executed": False,
                "analysis_scope": "static_python_ast",
            },
        ),
        BenchmarkCase(
            "class Example:\n    pass",
            required_metadata={
                "valid_syntax": True,
                "classes": 1,
                "executed": False,
            },
        ),
        BenchmarkCase(
            "import math\nfrom pathlib import Path",
            required_metadata={
                "valid_syntax": True,
                "imports": 2,
                "executed": False,
            },
        ),
        BenchmarkCase(
            "async def fetch():\n    return None",
            required_metadata={
                "valid_syntax": True,
                "functions": 1,
                "executed": False,
            },
        ),
        BenchmarkCase(
            "def broken(:\n    pass",
            required_metadata={
                "valid_syntax": False,
                "executed": False,
            },
        ),
        BenchmarkCase(
            "value = eval('1 + 1')",
            required_metadata={
                "valid_syntax": True,
                "review_required_calls": ["eval"],
                "executed": False,
            },
        ),
    ]

    report = SpecialistBenchmark().evaluate(
        ProgrammingSpecialist(), cases, minimum_accuracy=1.0
    )
    assert report.domain == "programming"
    assert report.total == 6
    assert report.accepted is True, report.case_results


def test_science_specialist_meets_initial_domain_benchmark():
    from aviy.models.specialists import (
        BenchmarkCase,
        ScienceSpecialist,
        SpecialistBenchmark,
    )

    cases = [
        BenchmarkCase(
            "What causes this observed effect?",
            required_metadata={
                "validated_claim": False,
                "question_type": "causal",
                "evidence_status": "not_supplied",
            },
            expected_confidence=0.0,
            expected_evidence_required=True,
        ),
        BenchmarkCase(
            "Does this hypothesis explain the data?",
            required_metadata={
                "validated_claim": False,
                "question_type": "hypothesis",
                "evidence_status": "not_supplied",
            },
            expected_confidence=0.0,
            expected_evidence_required=True,
        ),
        BenchmarkCase(
            "Explain the process of photosynthesis.",
            required_metadata={
                "validated_claim": False,
                "question_type": "explanatory",
                "evidence_status": "not_supplied",
            },
            expected_confidence=0.0,
            expected_evidence_required=True,
        ),
        BenchmarkCase(
            "What is the effect of temperature on reaction rate?",
            required_metadata={
                "validated_claim": False,
                "question_type": "causal",
                "evidence_status": "not_supplied",
            },
            expected_confidence=0.0,
            expected_evidence_required=True,
        ),
        BenchmarkCase(
            "Evaluate this scientific theory.",
            required_metadata={
                "validated_claim": False,
                "question_type": "hypothesis",
                "evidence_status": "not_supplied",
            },
            expected_confidence=0.0,
            expected_evidence_required=True,
        ),
        BenchmarkCase(
            "Describe the structure of a cell.",
            required_metadata={
                "validated_claim": False,
                "question_type": "explanatory",
                "evidence_status": "not_supplied",
            },
            expected_confidence=0.0,
            expected_evidence_required=True,
        ),
    ]

    report = SpecialistBenchmark().evaluate(
        ScienceSpecialist(), cases, minimum_accuracy=1.0
    )
    assert report.domain == "science"
    assert report.total == 6
    assert report.accepted is True, report.case_results


def test_specialist_benchmark_validates_confidence_and_evidence_criteria():
    import pytest

    from aviy.models.specialists import (
        BenchmarkCase,
        MathematicsSpecialist,
        SpecialistBenchmark,
    )

    invalid_cases = [
        BenchmarkCase("2 + 2", expected_numeric=4, expected_confidence=1.5),
        BenchmarkCase("2 + 2", expected_numeric=4, expected_confidence=float("nan")),
        BenchmarkCase(
            "2 + 2", expected_numeric=4,
            expected_evidence_required=cast(Any, "yes"),
        ),
    ]

    for case in invalid_cases:
        with pytest.raises(ValueError):
            SpecialistBenchmark().evaluate(
                MathematicsSpecialist(), [case], minimum_accuracy=1.0
            )


def test_specialist_router_routes_each_supported_domain():
    from aviy.models.specialists import (
        MathematicsSpecialist,
        ProgrammingSpecialist,
        ScienceSpecialist,
        SpecialistRouter,
    )

    router = SpecialistRouter(
        [
            MathematicsSpecialist(),
            ProgrammingSpecialist(),
            ScienceSpecialist(),
        ]
    )

    assert [item.domain for item in router.route("solve this equation")] == [
        "mathematics"
    ]
    assert [item.domain for item in router.route("review this Python code")] == [
        "programming"
    ]
    assert [item.domain for item in router.route("explain physics")] == ["science"]


def test_specialist_router_uses_all_specialists_for_unmatched_prompt():
    from aviy.models.specialists import (
        MathematicsSpecialist,
        ProgrammingSpecialist,
        ScienceSpecialist,
        SpecialistRouter,
    )

    router = SpecialistRouter(
        [
            MathematicsSpecialist(),
            ProgrammingSpecialist(),
            ScienceSpecialist(),
        ]
    )

    selected = router.route("Please examine this unfamiliar topic.")
    assert {item.domain for item in selected} == {
        "mathematics",
        "programming",
        "science",
    }


def test_specialist_collaboration_collects_all_routed_results():
    from aviy.models.specialists import (
        MathematicsSpecialist,
        ProgrammingSpecialist,
        ScienceSpecialist,
        SpecialistRouter,
    )

    router = SpecialistRouter(
        [
            MathematicsSpecialist(),
            ProgrammingSpecialist(),
            ScienceSpecialist(),
        ]
    )

    results = router.collaborate("Please examine this unfamiliar topic.")

    assert len(results) == 3
    assert {result.specialist for result in results} == {
        "mathematics",
        "programming",
        "science",
    }


def test_research_engine_rejects_conflicting_duplicate_source_ids():
    first_content = "Original finding."
    second_content = "Different finding."
    first = ResearchSource(
        "same-id",
        "https://example.com/one",
        "First",
        "MIT",
        first_content,
        hashlib.sha256(first_content.encode()).hexdigest(),
        False,
        "Unverified",
    )
    second = ResearchSource(
        "same-id",
        "https://example.com/two",
        "Second",
        "MIT",
        second_content,
        hashlib.sha256(second_content.encode()).hexdigest(),
        False,
        "Unverified",
    )
    engine = ResearchEngine(
        discover=lambda query: [first, second],
        verify=lambda item: True,
        allowed_licenses={"MIT"},
    )

    with pytest.raises(ValueError, match="conflicting"):
        engine.run("research question")


def test_research_engine_rejects_citation_without_identity():
    content = "Finding."
    source = ResearchSource(
        "source-1",
        "https://example.com/data",
        "Data",
        "MIT",
        content,
        hashlib.sha256(content.encode()).hexdigest(),
        False,
        "Unverified",
        citations=(Citation("", "https://example.com/paper", "Paper"),),
    )
    engine = ResearchEngine(
        discover=lambda query: [source],
        verify=lambda item: True,
        allowed_licenses={"MIT"},
    )

    with pytest.raises(ValueError, match="citation"):
        engine.run("research question")


def test_research_engine_requires_boolean_verification_result():
    content = "Finding."
    source = ResearchSource(
        "source-1",
        "https://example.com/data",
        "Data",
        "MIT",
        content,
        hashlib.sha256(content.encode()).hexdigest(),
        False,
        "Unverified",
    )
    engine = ResearchEngine(
        discover=lambda query: [source],
        verify=cast(Any, lambda item: "false"),
        allowed_licenses={"MIT"},
    )

    with pytest.raises(TypeError, match="Boolean"):
        engine.run("research question")


@pytest.mark.parametrize(
    ("overrides", "exception"),
    [
        ({"source_id": None}, ValueError),
        ({"title": 123}, ValueError),
        ({"license": None}, ValueError),
        ({"content": "not bytes"}, TypeError),
        ({"citations": []}, TypeError),
        ({"publisher": None}, TypeError),
        ({"retrieved_at": "not-a-date"}, ValueError),
        ({"retrieved_at": "2026-01-01T00:00:00"}, ValueError),
    ],
)
def test_phase18_rejects_malformed_acquisition_metadata(overrides, exception):
    arguments = {
        "source_id": "source-1",
        "uri": "https://example.com/data",
        "title": "Research data",
        "license": "MIT",
        "content": b"content",
    }
    arguments.update(overrides)
    with pytest.raises(exception):
        make_acquirer().acquire(**arguments)


def test_phase18_rejects_invalid_citation_record():
    with pytest.raises(ValueError, match="Citation"):
        make_acquirer().acquire(
            source_id="source-1",
            uri="https://example.com/data",
            title="Research data",
            license="MIT",
            content=b"content",
            citations=cast(Any, (object(),)),
        )


def test_phase18_rejects_malformed_discovery_results():
    engine = ResearchEngine(
        discover=cast(Any, lambda query: [object()]),
        verify=lambda source: True,
        allowed_licenses={"MIT"},
    )
    with pytest.raises(TypeError, match="source record"):
        engine.run("research question")


def test_phase18_rejects_non_sequence_discovery_results():
    engine = ResearchEngine(
        discover=cast(Any, lambda query: iter(())),
        verify=lambda source: True,
    )
    with pytest.raises(TypeError, match="list or tuple"):
        engine.run("research question")


def test_phase18_rejects_malformed_source_hash():
    source = ResearchSource(
        "source-1",
        "https://example.com/data",
        "Data",
        "MIT",
        "content",
        "not-a-hash",
        False,
        "Unverified",
    )
    engine = ResearchEngine(
        discover=lambda query: [source],
        verify=lambda item: True,
        allowed_licenses={"MIT"},
    )
    with pytest.raises(ValueError, match="hash"):
        engine.run("research question")


def test_phase18_rejects_non_boolean_verifier_result():
    content = "content"
    source = ResearchSource(
        "source-1",
        "https://example.com/data",
        "Data",
        "MIT",
        content,
        hashlib.sha256(content.encode("utf-8")).hexdigest(),
        False,
        "Unverified",
    )
    engine = ResearchEngine(
        discover=lambda query: [source],
        verify=cast(Any, lambda item: "yes"),
        allowed_licenses={"MIT"},
    )
    with pytest.raises(TypeError, match="Boolean"):
        engine.run("research question")


@pytest.mark.parametrize(
    "overrides",
    [
        {"minimum_gain": True},
        {"minimum_gain": float("nan")},
        {"maximum_regression": float("inf")},
        {"reproducibility_tolerance": -1},
        {"metric": ""},
        {"name": " "},
        {"direction": "increase"},
    ],
)
def test_phase19_rejects_invalid_objective_values(overrides):
    values = {
        "name": "accuracy",
        "metric": "accuracy",
        "minimum_gain": 0.05,
        "maximum_regression": 0.0,
        "direction": "maximize",
        "reproducibility_tolerance": 0.001,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        ImprovementObjective(**values)


# Additional hardening regression tests


def test_candidate_runner_rejects_non_boolean_isolation_flag(tmp_path):
    class InvalidIsolationBackend(TestOnlyExperimentBackend):
        isolation_verified = cast(Any, 1)

    runner = CandidateExperimentRunner(
        tmp_path, cast(Any, InvalidIsolationBackend())
    )
    with pytest.raises(RuntimeError, match="isolation"):
        runner.evaluate(
            "candidate-invalid-isolation",
            (make_objective(),),
            lambda workspace: {"accuracy": 0.5},
            lambda workspace: {"accuracy": 0.8},
            repeats=2,
        )


def test_candidate_runner_rejects_invalid_metric_names(tmp_path):
    runner = CandidateExperimentRunner(tmp_path, TestOnlyExperimentBackend())
    with pytest.raises(ValueError, match="metric scores"):
        runner.evaluate(
            "candidate-invalid-metric",
            (make_objective(),),
            lambda workspace: {"": 0.5},
            lambda workspace: {"accuracy": 0.8},
            repeats=2,
        )


def test_candidate_runner_rejects_non_json_configuration(tmp_path):
    runner = CandidateExperimentRunner(tmp_path, TestOnlyExperimentBackend())
    with pytest.raises(ValueError, match="JSON"):
        runner.evaluate(
            "candidate-invalid-config",
            (make_objective(),),
            lambda workspace: {"accuracy": 0.5},
            lambda workspace: {"accuracy": 0.8},
            repeats=2,
            configuration={"not_finite": float("nan")},
        )


def test_research_engine_rejects_non_string_question():
    engine = ResearchEngine(lambda query: [], lambda source: True)
    with pytest.raises(ValueError, match="question"):
        engine.run(cast(Any, None))


def test_research_engine_rejects_non_boolean_source_verified_flag():
    content = "research evidence"
    source = ResearchSource(
        "source-bad-verified",
        "https://example.com/evidence",
        "Evidence",
        "MIT",
        content,
        hashlib.sha256(content.encode()).hexdigest(),
        cast(Any, 1),
        "Unverified",
    )
    engine = ResearchEngine(lambda query: [source], lambda item: True)
    with pytest.raises(ValueError, match="metadata"):
        engine.run("research question")


def test_update_manager_rejects_non_boolean_signature_verification(tmp_path):
    manager = make_update_manager(
        tmp_path,
        verifier=lambda payload, signature: 1,
    )
    with pytest.raises(PermissionError, match="signature"):
        manager.verify(make_manifest())


def test_update_manager_preserves_backup_when_rollback_fails(
    tmp_path,
    monkeypatch,
):
    import os
    from pathlib import Path

    manager = make_update_manager(tmp_path)
    first = tmp_path / "component" / "first.txt"
    second = tmp_path / "component" / "second.txt"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"old first")
    second.write_bytes(b"old second")

    files = (
        UpdateFile(
            "component/first.txt",
            b"new first",
            hashlib.sha256(b"new first").hexdigest(),
        ),
        UpdateFile(
            "component/second.txt",
            b"new second",
            hashlib.sha256(b"new second").hexdigest(),
        ),
    )
    unsigned = UpdateManifest(
        "update-rollback",
        "core",
        "1.0.2",
        files,
        b"test-signature",
        b"",
    )
    manifest = UpdateManifest(
        unsigned.update_id,
        unsigned.component,
        unsigned.version,
        unsigned.files,
        unsigned.signature,
        canonical_manifest_payload(unsigned),
    )

    real_replace = os.replace
    calls = 0

    def fail_install_and_rollback(source, destination):
        nonlocal calls
        calls += 1
        if calls in (2, 3):
            raise OSError("injected replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "aviy.infrastructure.updates.os.replace",
        fail_install_and_rollback,
    )

    with pytest.raises(RuntimeError, match="Recovery backup preserved at:") as error:
        manager.apply(
            manifest,
            compatibility_check=lambda item: True,
            approval=lambda item: True,
        )

    backup_text = str(error.value).split("Recovery backup preserved at: ", 1)[1]
    backup = Path(backup_text)
    assert backup.is_dir()
    assert (backup / "component" / "first.txt").read_bytes() == b"old first"
    assert first.read_bytes() == b"new first"
    assert calls == 3


def test_phase19_adoption_ledger_requires_eligible_candidate_and_approval(tmp_path):
    runner = CandidateExperimentRunner(
        tmp_path / "experiments",
        TestOnlyExperimentBackend(),
    )
    result = runner.evaluate(
        "candidate-ledger-approved",
        (make_objective(),),
        lambda workspace: {"accuracy": 0.5},
        lambda workspace: {"accuracy": 0.8},
        repeats=3,
    )

    ledger = CandidateAdoptionLedger(tmp_path / "adoption" / "decisions.jsonl")
    digest = ledger.record_decision(result, approved=True)

    assert len(digest) == 64
    assert ledger.verify() is True

    with pytest.raises(ValueError, match="already exists"):
        ledger.record_decision(result, approved=True)


def test_phase19_adoption_ledger_rejects_ineligible_candidate(tmp_path):
    runner = CandidateExperimentRunner(
        tmp_path / "experiments",
        TestOnlyExperimentBackend(),
    )
    result = runner.evaluate(
        "candidate-ledger-rejected",
        (make_objective(),),
        lambda workspace: {"accuracy": 0.5},
        lambda workspace: {"accuracy": 0.51},
        repeats=2,
    )
    assert result.adopted_eligible is False

    ledger = CandidateAdoptionLedger(tmp_path / "decisions.jsonl")
    with pytest.raises(ValueError, match="ineligible"):
        ledger.record_decision(result, approved=True)

    ledger.record_decision(result, approved=False)
    assert ledger.verify() is True


def test_phase19_adoption_ledger_detects_tampering(tmp_path):
    runner = CandidateExperimentRunner(
        tmp_path / "experiments",
        TestOnlyExperimentBackend(),
    )
    result = runner.evaluate(
        "candidate-ledger-tamper",
        (make_objective(),),
        lambda workspace: {"accuracy": 0.5},
        lambda workspace: {"accuracy": 0.8},
        repeats=2,
    )

    path = tmp_path / "decisions.jsonl"
    ledger = CandidateAdoptionLedger(path)
    ledger.record_decision(result, approved=True)

    content = path.read_text(encoding="utf-8")
    path.write_text(
        content.replace('"approved":true', '"approved":false'), encoding="utf-8"
    )

    assert ledger.verify() is False


def test_phase19_windows_sandbox_backend_never_executes_host_callback(tmp_path):
    from aviy.agents.runtime import SandboxToolDefinition
    from aviy.autonomous.self_improvement import WindowsSandboxExperimentBackend

    class FakeExecutor:
        isolation_verified = True

        def __init__(self):
            self.calls = []

        def execute(self, tool, arguments, *, timeout_seconds):
            self.calls.append(tool.name)
            return {"accuracy": 0.5 if tool.name == "baseline" else 0.8}

    executor = FakeExecutor()
    baseline = SandboxToolDefinition(
        name="baseline",
        script="param($Arguments) @{ accuracy = 0.5 }",
        required_permission="sandbox.execute",
    )
    candidate = SandboxToolDefinition(
        name="candidate",
        script="param($Arguments) @{ accuracy = 0.8 }",
        required_permission="sandbox.execute",
    )
    backend = WindowsSandboxExperimentBackend(executor, baseline, candidate)

    def forbidden_host_callback(_workspace):
        raise AssertionError("The host callback must never execute.")

    result = backend.run(
        forbidden_host_callback,
        tmp_path,
        role="candidate",
        repeat=0,
    )

    assert result == {"accuracy": 0.8}
    assert executor.calls == ["candidate"]
