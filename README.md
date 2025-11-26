# StockFlow

A fully automated machine learning pipeline for stock market prediction and analysis, built with Apache Airflow, MLflow, and containerized microservices. **Features complete CI/CD automation** with intelligent deployment strategies based on code changes.

## ✨ Key Features

- 🚀 **Fully Automated CI/CD Pipeline** - Push code and let GitHub Actions handle the rest
- 🐳 **Smart Container Builds** - Automatic detection and building of only changed containers
- ✅ **DAG Validation** - Automatic Airflow DAG syntax validation on every commit
- 🔄 **Intelligent Deployment** - Auto-deploy containers and sync DAGs based on what changed
- 📊 **End-to-End ML Pipeline** - From data ingestion to model serving
- 🔒 **Production-Ready** - Built-in health checks, monitoring, and security

## 🤖 Automation Pipeline

StockFlow uses a **two-track automation strategy** that processes changes based on what was modified:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Push to main branch                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ┌───────────────────┐       ┌───────────────────┐
        │  Container Changes │       │    DAG Changes    │
        │   containers/**   │       │  airflow/dags/**  │
        └───────────────────┘       └───────────────────┘
                    │                           │
                    ▼                           ▼
        ┌───────────────────┐       ┌───────────────────┐
        │   Detect Changed  │       │  Validate DAGs    │
        │    Containers     │       │  • Syntax check   │
        └───────────────────┘       │  • Import test    │
                    │               │  • Structure check│
                    ▼               └───────────────────┘
        ┌───────────────────┐                   │
        │  Build & Push to  │                   ▼
        │      ghcr.io      │       ┌───────────────────┐
        └───────────────────┘       │  Sync to Airflow  │
                    │               │     Server        │
                    ▼               └───────────────────┘
        ┌───────────────────┐
        │ Deploy via        │
        │ Portainer Webhook │
        └───────────────────┘
                    │
                    ▼
        ┌───────────────────┐
        │  Health Check     │
        │  Validation       │
        └───────────────────┘
```

### Container Changes → Build & Deploy

When you modify files in `containers/`:

1. **Smart Detection** - Only builds containers that actually changed
2. **Multi-Architecture Build** - Creates `linux/amd64` and `linux/arm64` images
3. **Registry Push** - Pushes to GitHub Container Registry (`ghcr.io`)
4. **Auto-Deploy** - Triggers Portainer webhook for production deployment
5. **Health Validation** - Runs health checks to verify deployment success

### DAG Changes → Validate & Sync

When you modify files in `airflow/dags/`:

1. **Syntax Validation** - Checks Python syntax using AST parsing
2. **Import Testing** - Verifies DAGs can be imported without errors
3. **Structure Check** - Ensures proper DAG definitions exist
4. **Auto-Sync** - Deploys validated DAGs to production Airflow server via rsync

## 🏗️ Architecture

```
                                    ┌─────────────────┐
                                    │   MinIO (S3)    │
                                    │  Object Storage │
                                    └────────┬────────┘
                                             │
┌──────────────────────────────────────────────────────────────────────┐
│                         Apache Airflow                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  ETL DAG    │───▶│  Train DAG  │───▶│  Monitor    │              │
│  │  Pipeline   │    │  Pipeline   │    │    DAG      │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
└──────────────────────────────────────────────────────────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  ETL Container  │ │ Train Container │ │   Monitoring    │
│  Bronze → Silver│ │  MLflow Models  │ │   Container     │
│  Silver → Gold  │ └─────────────────┘ └─────────────────┘
└─────────────────┘          │
                             ▼
                    ┌─────────────────┐
                    │     MLflow      │
                    │ Tracking Server │
                    └────────┬────────┘
                             │
         ┌───────────────────┴───────────────────┐
         ▼                                       ▼
┌─────────────────┐                     ┌─────────────────┐
│ Serve Container │                     │  App Container  │
│ FastAPI Model   │◀────────────────────│  Streamlit UI   │
│    Serving      │      /predict       │                 │
└─────────────────┘                     └─────────────────┘
```

### Components

| Component | Description | Technology |
|-----------|-------------|------------|
| **ETL** | Data pipeline using Medallion Architecture (Bronze → Silver → Gold) | Python, Pandas |
| **Train** | Model training with experiment tracking | Python, MLflow |
| **Serve** | Production model serving with API | FastAPI, MLflow |
| **App** | User interface for predictions | Streamlit |
| **Monitoring** | System and model monitoring | Prometheus/Grafana |

## 📋 Prerequisites

- **Docker** & **Docker Compose** v2.0+
- **Python** 3.11+
- **Task** (optional, for running build commands)
- **Git** for version control

### Required Services

- **PostgreSQL** - MLflow backend store
- **MinIO** (or S3) - Artifact storage
- **Airflow** - Workflow orchestration (production)
- **Portainer** - Container management (production)

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Encall/stockflow.git
cd stockflow
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your configuration
vim .env
```

Required environment variables:

```bash
# PostgreSQL
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=mlflow

# MLflow
MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:password@postgres:5432/mlflow
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://stockflow/mlflow-artifacts

# MinIO/S3
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_ENDPOINT_URL=https://your-minio-endpoint
```

### 3. Create Docker Network

```bash
docker network create stockflow-network
```

### 4. Start Core Services

```bash
docker compose up -d
```

### 5. Build Containers (Local Development)

```bash
# Using Task
task build-all

# Or manually
docker build -t stockflow:etl ./containers/etl
docker build -t stockflow:train ./containers/train
docker build -t stockflow:serve ./containers/serve
docker build -t stockflow:app ./containers/app
```

## 🔄 Development Workflow

### Making Changes

StockFlow's CI/CD pipeline automatically handles deployment based on what you change:

#### Example 1: Updating the Model Serving Logic

```bash
# Edit the serve container code
vim containers/serve/app/main.py

# Commit and push
git add .
git commit -m "feat(serve): add batch prediction endpoint"
git push origin main
```

**What happens automatically:**
1. ✅ Change detection identifies `containers/serve/**` was modified
2. 🐳 Docker image is built for `serve` container only
3. 📦 Image pushed to `ghcr.io/encall/stockflow/serve:latest`
4. 🚀 Portainer webhook triggers deployment
5. 🏥 Health check validates the deployment

#### Example 2: Adding a New DAG

```bash
# Create new DAG file
vim airflow/dags/my_new_dag.py

# Commit and push
git add .
git commit -m "feat(dags): add weekly retraining pipeline"
git push origin main
```

**What happens automatically:**
1. ✅ Change detection identifies `airflow/dags/**` was modified
2. 🔍 DAG validation runs (syntax, imports, structure)
3. 📤 Validated DAGs sync to production Airflow server
4. ✅ New DAG appears in Airflow UI

#### Example 3: Updating Multiple Components

```bash
# Edit multiple areas
vim containers/etl/src/gold.py
vim containers/train/src/model.py
vim airflow/dags/dags_pipeline.py

git add .
git commit -m "feat: add new feature engineering and model updates"
git push origin main
```

**What happens automatically:**
1. ✅ Both `containers/` and `airflow/dags/` changes detected
2. 🐳 ETL and Train containers built in parallel
3. 🔍 DAG validation runs concurrently
4. 📤 DAGs synced after validation passes
5. 🚀 Containers deployed after build completes

## 📦 Services

### ETL Container (`containers/etl/`)

Implements the Medallion Architecture for data processing:

- **Bronze Layer**: Raw data ingestion from MinIO
- **Silver Layer**: Data cleaning and validation
- **Gold Layer**: Feature engineering for ML models

```bash
# Run locally
cd containers/etl
uv sync
uv run python etl.py
```

### Train Container (`containers/train/`)

Model training with MLflow tracking:

- Experiment tracking and versioning
- Model registry integration
- Automated hyperparameter logging

### Serve Container (`containers/serve/`)

FastAPI-based model serving:

- **Endpoints**:
  - `GET /health` - Service health and loaded model info
  - `GET /metadata` - Model metadata
  - `POST /predict` - Generate predictions
  - `POST /reload` - Hot-reload latest production model

```bash
# Example prediction request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"feature1": 0.5, "feature2": 1.2}], "stock": "XLP"}'
```

### App Container (`containers/app/`)

Streamlit-based user interface for:

- Viewing predictions
- Exploring historical data
- Model performance dashboards

## 🔧 Configuration

### Container Configuration

Each container has its own `.env.example`:

```bash
containers/
├── etl/.env.example
├── train/.env.example
├── serve/.env.example
└── app/.env.example
```

### MLflow Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://mlflow:5000` |
| `MLFLOW_EXPERIMENT_NAMES` | Comma-separated experiment names | All experiments |
| `MLFLOW_PRODUCTION_TAG_KEY` | Tag key for production models | `production` |
| `MLFLOW_PRODUCTION_TAG_VALUE` | Tag value for production models | `true` |

## 🛠️ Project Structure

```
stockflow/
├── .github/
│   ├── scripts/
│   │   └── validate_dags.py       # DAG validation script
│   └── workflows/
│       ├── ci-cd.yml              # 🔄 Main orchestrator workflow
│       ├── docker-build.yml       # 🐳 Smart container builds
│       ├── deploy-app.yml         # 🚀 App deployment
│       ├── deploy-serve.yml       # 🚀 Serve deployment
│       ├── validate-dags.yml      # ✅ DAG validation
│       ├── sync-dags.yml          # 📤 DAG synchronization
│       └── commit-lint.yml        # 📝 Conventional commits
├── airflow/
│   └── dags/                      # Airflow DAG definitions
├── configs/                       # Configuration files
├── containers/
│   ├── app/                       # Streamlit application
│   ├── etl/                       # ETL pipeline
│   ├── monitoring/                # Monitoring stack
│   ├── serve/                     # Model serving API
│   └── train/                     # Model training
├── data/                          # Local data directory
├── docker-compose.yml             # Core services
├── Taskfile.yml                   # Task runner commands
└── README.md                      # This file
```

## 🎯 CI/CD Workflows Explained

### `ci-cd.yml` - Main Orchestrator

The primary workflow that coordinates all CI/CD activities:

- **Triggers**: Push to `main`, manual dispatch
- **Actions**: 
  - Detects which paths changed (`containers/` or `airflow/dags/`)
  - Calls appropriate sub-workflows based on changes
  - Ensures proper execution order (validate → build → deploy)

### `docker-build.yml` - Smart Container Builds

Intelligent container building:

- **Detection**: Uses `dorny/paths-filter` to identify changed containers
- **Matrix Build**: Builds only modified containers in parallel
- **Multi-Arch**: Creates AMD64 and ARM64 images
- **Registry**: Pushes to GitHub Container Registry with SHA and `latest` tags

### `deploy-app.yml` & `deploy-serve.yml` - Container Deployment

Production deployment via Portainer:

- **Trigger**: Called after successful container builds
- **Webhook**: Triggers Portainer service update
- **Health Check**: Validates deployment with retry logic

### `validate-dags.yml` - DAG Validation

Comprehensive DAG testing:

- **Syntax Check**: Python AST parsing
- **Import Test**: Verifies clean imports
- **Structure Check**: Ensures DAG definitions exist

### `sync-dags.yml` - DAG Synchronization

Secure DAG deployment:

- **Tailscale**: Secure VPN connection to production
- **Rsync**: Efficient file synchronization
- **Atomic**: Updates all DAGs in single operation

### `commit-lint.yml` - Commit Quality

Enforces Conventional Commits format on PR titles.

## 📊 Monitoring

### Health Checks

All containers include built-in health endpoints:

```bash
# Check serve container
curl http://localhost:8000/health

# Response
{
  "status": "healthy",
  "model_loaded": true,
  "run_id": "abc123...",
  "model_uri": "runs:/abc123.../model"
}
```

### Service Health

```bash
# Docker Compose health status
docker compose ps

# Container logs
docker compose logs -f serve
```

## 🔐 Security Notes

### Secrets Management

Required GitHub repository secrets for CI/CD:

| Secret | Purpose |
|--------|---------|
| `PORTAINER_WEBHOOK_URL` | App deployment webhook |
| `PORTAINER_SERVE_WEBHOOK_URL` | Serve deployment webhook |
| `APP_URL` | App health check URL |
| `SERVE_URL` | Serve health check URL |
| `TS_OAUTH_CLIENT_ID` | Tailscale OAuth client |
| `TS_OAUTH_SECRET` | Tailscale OAuth secret |
| `AIRFLOW_SERVER_IP` | Airflow server address |
| `AIRFLOW_SERVER_USER` | SSH user for Airflow |
| `AIRFLOW_SSH_PRIVATE_KEY` | SSH key for DAG sync |
| `AIRFLOW_DAGS_PATH` | DAG directory path |

### Best Practices

- Never commit secrets to the repository
- Use environment variables for configuration
- Rotate credentials regularly
- Use minimal permission principles for service accounts

## 📝 Commit Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

Types:
- feat:     New feature
- fix:      Bug fix
- docs:     Documentation
- style:    Code style (formatting, etc.)
- refactor: Code refactoring
- test:     Adding tests
- chore:    Maintenance tasks
- perf:     Performance improvements
- ci:       CI/CD changes
- build:    Build system changes
- revert:   Revert previous commit

Examples:
- feat(serve): add batch prediction endpoint
- fix(etl): handle missing dates correctly
- docs: update README with CI/CD info
- ci(docker): optimize multi-arch builds
```

## 👤 Author

**Encall**

- GitHub: [@Encall](https://github.com/Encall)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feat/amazing-feature`)
3. **Commit** your changes using conventional commits
4. **Push** to your branch (`git push origin feat/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow the existing code style
- Add tests for new features
- Update documentation as needed
- Ensure CI checks pass before requesting review

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

---

<p align="center">
  <b>StockFlow</b> - Automated ML Pipeline for Stock Prediction
  <br>
  <i>Push code. Let CI/CD handle the rest.</i>
</p>
