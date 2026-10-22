# Deploying to Google Cloud Run

The same Docker image runs in two Cloud Run services:

| Service | `APP_MODE` | What it is |
|---|---|---|
| `cattle-api` | `api` (default) | FastAPI: `/ask`, `/health` |
| `cattle-ui` | `ui` | Streamlit chat, calls `cattle-api` through `API_URL` |

Cloud Run sets `PORT`; `start.sh` passes it to uvicorn / Streamlit.

**Monitoring in the cloud is optional.** Without `DATABASE_URL` the API logs a
warning, keeps answering, and the chat hides the feedback buttons. To enable
it, point `DATABASE_URL` to a managed Postgres (e.g. Cloud SQL) — this has a
monthly cost, unlike the Cloud Run free tier. Ingestion (Kestra) keeps running
locally: the image ships the processed `data/processed/*.jsonl` files.

## 1. One-time setup

```bash
gcloud auth login
gcloud projects create cattle-assistant-<something-unique>   # or reuse a project
gcloud config set project <PROJECT_ID>
gcloud config set run/region us-central1
```

Link a billing account to the project in the console (Billing). Cloud Run
requires it even if usage stays within the free tier.

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

### Store the Groq key as a secret (never as a plain env var)

Console → Security → **Secret Manager** → *Create secret*: name `groq-api-key`,
value = your Groq key. Then allow Cloud Run's runtime account to read it:

```bash
gcloud projects describe <PROJECT_ID> --format="value(projectNumber)"
gcloud secrets add-iam-policy-binding groq-api-key --member="serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
```

## 2. Deploy the API

From the project folder (`--source .` builds the image in Cloud Build, not on
your machine; `.gcloudignore` keeps `.env` and `.venv` out of the upload):

```bash
gcloud run deploy cattle-api --source . --allow-unauthenticated --memory 1Gi --max-instances 1 --set-secrets "GROQ_API_KEY=groq-api-key:latest" --set-env-vars "GROQ_GENERATION_MODEL=openai/gpt-oss-120b"
```

Check it (use the URL printed by the command):

```bash
curl https://cattle-api-XXXX.us-central1.run.app/health
```

## 3. Deploy the chat UI

```bash
gcloud run services describe cattle-api --format="value(status.url)"
gcloud run deploy cattle-ui --source . --allow-unauthenticated --memory 1Gi --max-instances 1 --session-affinity --set-env-vars "APP_MODE=ui,API_URL=<API_URL_FROM_PREVIOUS_COMMAND>"
```

Open the URL printed for `cattle-ui`.

## 4. Updating

After adding papers (re-run ingestion locally) or changing code, re-run the two
`gcloud run deploy` commands. Model deprecated? Only the API needs it:

```bash
gcloud run services update cattle-api --update-env-vars "GROQ_GENERATION_MODEL=<new-model>"
```

## 5. Live demo tips

* Cold starts: with `--min-instances 0` the first request after a while can take
  10–20 s. For the event, keep one instance warm and revert afterwards:
  `gcloud run services update cattle-ui --min-instances 1` (and the same for
  `cattle-api`); later `--min-instances 0` to avoid idle costs.
* The services are public. `--max-instances 1` limits how much an unexpected
  traffic spike can consume (Cloud Run resources and your Groq quota).

## 6. Cost safeguards (public demo)

Three layers, from hardest to softest:

1. **`--max-instances 1`** per service: Cloud Run never runs more than one copy,
   whatever the traffic. With one API instance the question quota below is exact.
2. **Global question quota in the API** (`MAX_QUESTIONS_PER_MINUTE`, default 20;
   `MAX_QUESTIONS_PER_DAY`, default 300). Above it `/ask` answers **429** without
   calling the LLM. Change it without rebuilding:
   `gcloud run services update cattle-api --update-env-vars "MAX_QUESTIONS_PER_DAY=150"`
3. **Per-session limit in the chat** (`MAX_QUESTIONS_PER_SESSION`, default 15).

Plus a **budget alert** (Billing → Budgets & alerts), which notifies but does not stop spending.

Check what is active on a service:

```powershell
gcloud run services describe cattle-api --format=yaml | Select-String "Scale|MAX_QUESTIONS"
```

`autoscaling.knative.dev/maxScale` is the max-instances value; `minScale` the
instances kept warm.

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Permission 'iam.serviceaccounts.actAs' denied` | Your user needs `roles/iam.serviceAccountUser` on the runtime service account. |
| `Build failed` with a permissions message | Grant `roles/run.builder` to the build service account. |
| Container fails to start | `gcloud run services logs read cattle-api --limit 50` |
| `/ask` returns 503 "model not available" | Groq retired the model: update `GROQ_GENERATION_MODEL` (step 4). |
| Chat says "El servicio no responde" | Wrong `API_URL` in `cattle-ui`, or the API is cold-starting. Fix: `gcloud run services update cattle-ui --update-env-vars "API_URL=<api url>"` |
| `/ask` returns 429 | The question quota was reached; it resets after a minute / at 00:00 UTC. |

References: [Cloud Run troubleshooting](https://cloud.google.com/run/docs/troubleshooting),
[Cloud Run pricing](https://cloud.google.com/run/pricing).
