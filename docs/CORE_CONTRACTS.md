# Aviy Core Contracts

## Component interface

A component exposes a stable name and three lifecycle operations:

1. `initialize()` prepares its resources.
2. `start()` begins operation.
3. `stop()` releases resources.

Components should make lifecycle operations safe to call according
to their documented state requirements.

## Registry contract

A registry provides named registration and lookup.

- Names must be non-empty strings.
- Values cannot be `None`.
- Duplicate registration is rejected unless replacement is explicit.
- Missing names raise a registry exception.
- A frozen registry rejects modifications.
- Registration order is preserved.

## Dependency injection contract

The dependency container supports:

- Transient providers: a new instance is created for each resolution.
- Singleton providers: one instance is reused.
- Explicit instance binding.
- Duplicate binding protection.
- Circular dependency detection.
- Clear errors for unknown dependencies.

Providers receive the container and may resolve their own dependencies.

## Coordinator contract

The coordinator owns component lifecycle ordering.

- Components are registered before initialization.
- Initialization occurs in registration order.
- Startup occurs in registration order.
- Shutdown occurs in reverse order.
- Failed startup triggers cleanup.
- Lifecycle failures are reported with their original cause.
- A stopped coordinator cannot be restarted.

## Safety boundaries

These core contracts do not grant security privileges.
Authorization, resource limits, sandboxing and protected recovery
remain separate responsibilities.
