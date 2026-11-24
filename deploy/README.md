# Aviy local deployment

1. Install Python 3.12.
2. Create and activate a virtual environment.
3. Install Aviy and its development dependencies.
4. Set AVIY_TOKEN_SIGNING_KEY to a secret of at least 32 bytes.
5. Run `python -m aviy serve --host 127.0.0.1 --port 8000`.

Do not expose this initial deployment directly to a public network.
Production requires a managed secret store, TLS, monitoring, backups,
access controls and deployment-specific rollback.
