# Phase 25 - Security, Privacy and Governance Completion

## Implemented controls

- SQLite-backed local account records.
- PBKDF2-HMAC-SHA256 password hashing with per-account random salts.
- Constant-time password-hash and token-signature comparisons.
- HMAC-signed, expiring access tokens with persistent revocation.
- Role checks against current account state; disabling an account invalidates its tokens.
- Durable JSONL governance events with a keyed hash chain, sequence validation and restart-time integrity verification.
- The durable recovery API requires two distinct supplied approver names with the recovery-approver role. It checks these supplied claims but does not authenticate approver identities against the authentication service; callers must authenticate and authorize them before invoking recovery.
- Recovery results must be verified; failed recovery invokes rollback and records the outcome.
- A separate DualApproval policy checks two distinct Principal objects with the approver role; it is not automatically wired into the durable recovery API.

## Existing project security controls

Phase 25 extends the existing authorization policy, environment secret resolver,
HTTPS host policy, filesystem boundary and Windows Sandbox executor. The Windows
Sandbox remains the required OS-level isolation boundary for untrusted code.
The architecture experiment workspace is not a substitute for that sandbox.

## Deployment requirements and limitations

- The signing and ledger-integrity keys must be generated randomly, stored outside
  the database and ledger, and supplied through a protected secret manager.
- The SQLite database and ledger require restrictive OS permissions and protected
  backups. Windows deployments must configure suitable ACLs.
- This local authentication service is a project component, not a complete
  internet-facing identity provider. Production deployments still need secure
  transport, account enrollment and recovery procedures, rate limiting, monitoring,
  and operational key rotation.
- The durable JSONL ledger coordinates threads and instances within one Python
  process, but does not provide inter-process locking. Multi-process deployments
  need an external transactional append-only store or coordinated locking.
- A hash chain detects alteration when the integrity key remains protected; it
  does not prevent a privileged host administrator from deleting the ledger.
- The Windows Sandbox provides OS isolation only when the host supports it and
  its configuration is correctly enforced. Network egress and host ACLs remain
  deployment responsibilities.
- No claim is made that environment variables alone constitute a secrets vault.
