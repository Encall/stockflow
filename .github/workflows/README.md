# GitHub Actions Workflows

This directory contains the CI/CD workflows for the stockflow project.

## Workflows

### Main Pipeline (`ci-cd.yml`)
The primary orchestration workflow that:
- Detects changes in containers and Airflow DAGs
- Triggers appropriate sub-workflows based on changes
- Coordinates the full CI/CD pipeline

### Docker Build (`docker-build.yml`)
Builds and pushes Docker images to GitHub Container Registry (GHCR):
- Detects which containers have changed (etl, train, monitoring, app, serve)
- Builds multi-platform images (linux/amd64, linux/arm64)
- Tags images with SHA and promotes to semantic tags
- Outputs `app-built` and `serve-built` flags for deployment jobs

### Deployments

#### Deploy App (`deploy-app.yml`)
Automatically deploys the **app** container when built:
- Triggers Portainer webhook for app service
- Waits for deployment to complete
- Performs health check on app endpoint

**Required Secrets:**
- `PORTAINER_WEBHOOK_URL` - Portainer webhook URL for app service
- `APP_URL` - App URL for health check (e.g., `http://your-server:8501`)

#### Deploy Serve (`deploy-serve.yml`)
Automatically deploys the **serve** container when built:
- Triggers Portainer webhook for serve service
- Waits for deployment to complete (longer wait for model loading)
- Performs health check on `/health` endpoint
- Displays model serving status

**Required Secrets:**
- `PORTAINER_SERVE_WEBHOOK_URL` - Portainer webhook URL for serve service
- `SERVE_URL` - Serve URL for health check (e.g., `http://your-server:8000`)

### DAG Management

#### Validate DAGs (`validate-dags.yml`)
Validates Airflow DAGs for syntax and import errors before syncing.

#### Sync DAGs (`sync-dags.yml`)
Syncs validated DAGs to Airflow instance.

### Other Workflows

#### Commit Lint (`commit-lint.yml`)
Validates commit messages follow conventional commit format.

## Setup Instructions

### 1. Configure Portainer Webhooks

For both app and serve containers:

1. In Portainer, navigate to your service (Stacks or Containers)
2. Enable the webhook feature
3. Copy the webhook URL
4. Add as repository secret in GitHub:
   - Settings → Secrets and variables → Actions → New repository secret

### 2. Add Required Secrets

Add these secrets to your GitHub repository:

**For App Deployment:**
```
PORTAINER_WEBHOOK_URL=https://your-portainer-server:9443/api/webhooks/...
APP_URL=http://your-server:8501
```

**For Serve Deployment:**
```
PORTAINER_SERVE_WEBHOOK_URL=https://your-portainer-server:9443/api/webhooks/...
SERVE_URL=http://your-server:8000
```

### 3. Workflow Triggers

The CI/CD pipeline triggers on:
- **Push to main**: When changes are detected in `containers/**`, `airflow/dags/**`, or `.github/workflows/**`
- **Manual dispatch**: Can be triggered manually from Actions tab

## Workflow Flow

```
Push to main
    ↓
detect-changes
    ↓
├─→ validate-dags (if DAGs changed)
├─→ docker-build (if containers changed)
    ↓
    ├─→ deploy-app (if app built)
    └─→ deploy-serve (if serve built)
    ↓
└─→ sync-dags (if DAGs changed)
```

## Health Checks

### App Health Check
- Endpoint: `GET /`
- Expected: HTTP 200
- Retries: 3 times with 10s intervals

### Serve Health Check
- Endpoint: `GET /health`
- Expected: HTTP 200 with model status
- Retries: 5 times with 10s intervals (model loading can take time)
- Displays loaded model information on success

## Troubleshooting

### Deployment Skipped
If you see "⚠️ PORTAINER_*_WEBHOOK_URL secret not set", add the required secrets.

### Health Check Failed
- Verify the URL is accessible from GitHub Actions runners
- Check Portainer logs for deployment issues
- For serve: Allow more time for model loading
- Ensure the service is bound to `0.0.0.0` not `127.0.0.1`

### Build Not Triggering
- Check the `paths` filter in `ci-cd.yml`
- Verify changes are in the correct directories
- Check workflow run logs for change detection results
