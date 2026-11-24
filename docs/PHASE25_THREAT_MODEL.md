# Aviy Phase 25 - Threat Model and Governance Review

## Scope
This review covers authorization, runtime secret resolution, outbound network
policy, filesystem boundaries, and tamper-evident governance decisions.

## Assets
- Model weights, datasets, checkpoints and experiment outputs
- User data and credentials
- Runtime configuration and update manifests
- Audit and approval records

## Trust boundaries
1. Authenticated principal to permission enforcement.
2. Aviy process to external network destinations.
3. Aviy process to filesystem paths and symbolic links.
4. Runtime process to environment-provided secrets.
5. Candidate experiments and repair actions to production state.

## Threats and controls
| Threat | Control | Verification |
|---|---|---|
| Unauthorized operation | Explicit role-to-permission policy | Authorization tests |
| Secret committed or logged | Resolve secrets at runtime from environment | Resolver tests; no secret persistence API |
| SSRF or unsafe outbound destination | HTTPS host allowlist, public-address and port checks | Network policy tests |
| Path traversal or symlink escape | Root-confined path validation | Filesystem boundary tests |
| Audit history modification | Hash-linked governance ledger | Tamper-detection test |
| Single-person sensitive approval | Two distinct approver identities with the required role; durable recovery checks supplied identity and role claims | Dual-approval and recovery tests; caller authentication remains an integration responsibility |
| Unsafe automated repair | Named allowlisted actions and bounded execution | Self-healing tests |
| Unreproducible architecture adoption | Repeat experiments and baseline gates | Evolution tests |

## Residual risks and deployment requirements
- Environment variables are a runtime integration boundary, not a dedicated
  hardware-backed secret vault. Production deployments should use an approved
  operating-system or cloud secret manager.
- Host allowlists must be maintained by an administrator. DNS rebinding and
  proxy behavior require deployment-level egress controls.
- The durable recovery API checks two distinct supplied names and role claims,
  but does not authenticate those identities against the authentication
  service. Callers must authenticate and authorize approvers before invoking
  recovery. The separate DualApproval policy checks Principal objects but
  is not automatically integrated with the durable recovery API.
- Filesystem validation must be combined with restrictive OS permissions.
- The durable governance ledger is an append-only JSONL file with an HMAC
  hash chain and restart-time integrity verification. Its lock coordinates
  threads and ledger instances only within one Python process; multi-process
  writers require external coordination. A privileged host administrator
  can still delete or replace the ledger, so independent external retention
  remains necessary. A separate in-memory governance ledger also exists.
- The experiment workspace is isolated by temporary directories, not by a
  security sandbox. Untrusted candidate code must run through the existing
  Windows Sandbox or another OS-enforced sandbox before execution.
- Resource metrics may be partial when optional platform telemetry libraries
  or GPU drivers are unavailable.

## Review conclusion
The implementation provides testable application-level boundaries. It does
not claim to replace operating-system isolation, external audit retention,
hardware-backed secret storage, or administrator-managed deployment controls.
Those deployment controls remain mandatory for production use.
