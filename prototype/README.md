# MATMED prototype deployment

The Worker serves the dashboard and stores small progress reports in Workers KV.
Colab executes PyTorch. This is not a persistent inference service or a validated
drug-discovery product. No dataset rows, model weights, or generated molecules
are transmitted by the progress connector.

## Deploy

Use Node.js 22 or newer. `artifacts/matmed_prototype.zip` contains this complete
deployment project without dependencies, credentials or local KV data.

```sh
cd prototype
npm ci
npx wrangler login
npx wrangler kv namespace create RUNS
```

Replace `REPLACE_WITH_KV_NAMESPACE_ID` in `wrangler.jsonc` with the returned ID.
Then set two DIFFERENT randomly generated secrets (at least 32 random bytes each):

```sh
npx wrangler secret put WRITE_TOKEN
npx wrangler secret put READ_TOKEN
npm run deploy
```

The dashboard is served at the Worker URL. Enter only READ_TOKEN in its password
field. The token is retained in memory, not local storage. It can read all reports
in this single-project instance; this is not multi-tenant authorization.

In Colab Secrets set MATMED_API_URL to the Worker URL and MATMED_WRITE_TOKEN to
WRITE_TOKEN. Enable notebook access and set CONNECT_DASHBOARD=True in the notebook.
Keep secrets out of Git, screenshots, notebook outputs and frontend builds.
The API rejects all requests if the relevant secret is not configured.

## API contract

- `GET /api/health`: public health only.
- `PUT /api/runs/:run_id`: Bearer WRITE_TOKEN; schema-version-1 JSON report.
- `GET /api/runs`: Bearer READ_TOKEN; paginated summaries (`cursor`).
- `GET /api/runs/:run_id`: Bearer READ_TOKEN; complete report.

Run IDs use `matmed-` plus letters, digits and hyphens. Reports are limited to
256 KiB. Write a single sequential stream per run ID. KV is eventually consistent:
the dashboard may briefly show older data after upload. The dashboard polls every
15 seconds only while its tab is visible. Do not send patient data or secrets.
For a public multi-user product, add account-scoped authorization and rate limiting
before sharing read tokens broadly.

ALLOWED_ORIGIN defaults to the Worker's own origin. Set it only if an external
frontend must read this API. CORS is not a replacement for bearer authentication.

## Local tests

```sh
npm test
node tests/local-server.mjs
```

The integration harness listens on 127.0.0.1:8787, uses in-memory storage and test
credentials `local-read` / `local-write`; it must never be deployed. Configure:

```sh
MATMED_API_URL=http://127.0.0.1:8787 MATMED_WRITE_TOKEN=local-write \
  ../.venv/bin/python ../validity_pipeline.py --mode smoke --output ../runs/connected
```

Open localhost:8787 and enter local-read. The Worker handler is the same as production,
but the harness does NOT test actual Cloudflare KV or deployment behavior.

Cloudflare references:
- https://developers.cloudflare.com/workers/wrangler/configuration/
- https://developers.cloudflare.com/kv/concepts/kv-bindings/
- https://developers.cloudflare.com/workers/configuration/environment-variables/
