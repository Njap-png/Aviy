"""Tests for core interfaces, registries, DI and lifecycle coordination."""

import pytest

from aviy.core import (
    CircularDependencyError,
    Container,
    Coordinator,
    DependencyResolutionError,
    DuplicateRegistrationError,
    LifecycleError,
    LifecycleState,
    Lifetime,
    Registry,
    RegistryError,
    UnknownRegistrationError,
)


def test_registry_register_and_lookup():
    registry = Registry[str]()
    registry.register("alpha", "value")
    assert registry.get("alpha") == "value"
    assert registry.names == ("alpha",)
    assert len(registry) == 1


def test_registry_rejects_duplicate():
    registry = Registry[str]()
    registry.register("alpha", "first")
    with pytest.raises(DuplicateRegistrationError):
        registry.register("alpha", "second")


def test_registry_allows_explicit_replacement():
    registry = Registry[str]()
    registry.register("alpha", "first")
    registry.register("alpha", "second", replace=True)
    assert registry.get("alpha") == "second"


def test_registry_rejects_invalid_name_and_none():
    registry = Registry()
    with pytest.raises(ValueError):
        registry.register("", object())
    with pytest.raises(ValueError):
        registry.register("alpha", None)


def test_registry_unknown_lookup():
    with pytest.raises(UnknownRegistrationError):
        Registry().get("missing")


def test_registry_factory_creation():
    registry = Registry()
    registry.register("number", lambda value: value * 2)
    assert registry.create("number", 21) == 42


def test_registry_rejects_non_callable_creation():
    registry = Registry()
    registry.register("value", "text")
    with pytest.raises(RegistryError):
        registry.create("value")


def test_registry_freeze_blocks_changes():
    registry = Registry()
    registry.register("alpha", 1)
    registry.freeze()
    assert registry.frozen
    with pytest.raises(RegistryError):
        registry.register("beta", 2)
    with pytest.raises(RegistryError):
        registry.unregister("alpha")


def test_container_transient_lifetime():
    container = Container()
    container.bind("object", lambda _: object())
    assert container.resolve("object") is not container.resolve("object")


def test_container_singleton_lifetime():
    container = Container()
    container.bind("object", lambda _: object(), lifetime=Lifetime.SINGLETON)
    assert container.resolve("object") is container.resolve("object")


def test_container_instance_binding():
    container = Container()
    instance = object()
    container.bind_instance("object", instance)
    assert container.resolve("object") is instance


def test_container_rejects_duplicate_binding():
    container = Container()
    container.bind("value", lambda _: 1)
    with pytest.raises(DuplicateRegistrationError):
        container.bind("value", lambda _: 2)


def test_container_unknown_dependency():
    with pytest.raises(DependencyResolutionError):
        Container().resolve("missing")


def test_container_detects_circular_dependency():
    container = Container()
    container.bind("a", lambda c: c.resolve("b"))
    container.bind("b", lambda c: c.resolve("a"))
    with pytest.raises(CircularDependencyError):
        container.resolve("a")


def test_container_rejects_none_provider_result():
    container = Container()
    container.bind("empty", lambda _: None)
    with pytest.raises(DependencyResolutionError):
        container.resolve("empty")


class DummyComponent:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def initialize(self):
        self.events.append(f"initialize:{self.name}")

    def start(self):
        self.events.append(f"start:{self.name}")

    def stop(self):
        self.events.append(f"stop:{self.name}")


def test_coordinator_lifecycle_order():
    events = []
    coordinator = Coordinator()
    coordinator.add(DummyComponent("one", events))
    coordinator.add(DummyComponent("two", events))

    coordinator.initialize()
    assert coordinator.state is LifecycleState.INITIALIZED

    coordinator.start()
    assert coordinator.state is LifecycleState.RUNNING

    coordinator.shutdown()
    assert coordinator.state is LifecycleState.STOPPED

    assert events == [
        "initialize:one",
        "initialize:two",
        "start:one",
        "start:two",
        "stop:two",
        "stop:one",
    ]


def test_coordinator_rejects_duplicate_components():
    coordinator = Coordinator()
    coordinator.add(DummyComponent("one", []))
    with pytest.raises(DuplicateRegistrationError):
        coordinator.add(DummyComponent("one", []))


def test_coordinator_rejects_add_after_initialization():
    coordinator = Coordinator()
    coordinator.initialize()
    with pytest.raises(LifecycleError):
        coordinator.add(DummyComponent("late", []))


def test_coordinator_rejects_start_before_initialize():
    coordinator = Coordinator()
    with pytest.raises(LifecycleError):
        coordinator.start()


def test_coordinator_shutdown_before_initialize():
    coordinator = Coordinator()
    coordinator.shutdown()
    assert coordinator.state is LifecycleState.STOPPED


def test_coordinator_start_failure_cleans_up():
    events = []

    class FailingComponent(DummyComponent):
        def start(self):
            events.append(f"start:{self.name}")
            raise RuntimeError("expected test failure")

    coordinator = Coordinator()
    coordinator.add(DummyComponent("first", events))
    coordinator.add(FailingComponent("second", events))

    coordinator.initialize()
    with pytest.raises(LifecycleError):
        coordinator.start()

    assert coordinator.state is LifecycleState.FAILED
    assert "stop:second" in events
    assert "stop:first" in events


def test_coordinator_cannot_restart_after_shutdown():
    coordinator = Coordinator()
    coordinator.initialize()
    coordinator.shutdown()
    with pytest.raises(LifecycleError):
        coordinator.initialize()
