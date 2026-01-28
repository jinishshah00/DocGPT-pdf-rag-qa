#!/usr/bin/env bash
# deploy/fly_deploy.sh
# Helper script with commands to deploy backend, postgres, and frontend to Fly.io.
# This script does not install flyctl. Run it step-by-step or copy commands.
set -e

echo "1) Install flyctl if needed: https://fly.io/docs/hands-on/install-flyctl/"

echo "2) Login to Fly"
echo "   flyctl auth login"

echo "3) Create backend app"
echo "   (run from repo root or backend dir)
   flyctl apps create docgpt-backend || true"

echo "4) Create Postgres (managed)"
echo "   flyctl postgres create --name docgpt-db --region <your-region>"

echo "5) Attach Postgres to backend app and get DATABASE_URL"
echo "   FLY_POSTGRES=$(flyctl postgres attach --app docgpt-backend docgpt-db)"
echo "   # or use flyctl postgres connect details to set DATABASE_URL as a secret"

echo "6) Set secrets for backend (example):"
echo "   flyctl secrets set OPENAI_API_KEY=sk-... JWT_SECRET=change-me DATABASE_URL=postgres://..."

echo "7) Deploy backend (from repo root):"
echo "   # build with Dockerfile in infra/backend.Dockerfile or using flyctl build"
echo "   flyctl deploy --app docgpt-backend --config deploy/fly_backend.toml.template"

echo "8) Create frontend app and deploy"
echo "   flyctl apps create docgpt-frontend || true"
echo "   flyctl secrets set API_BASE_PUBLIC=https://docgpt-backend.fly.dev"
echo "   flyctl deploy --app docgpt-frontend --config deploy/fly_frontend.toml.template"

echo "Notes:
- Edit the template fly.toml files or run 'flyctl launch' in each folder to generate a proper fly.toml.
- Ensure you set all required secrets: OPENAI_API_KEY, JWT_SECRET, TAVILY_API_KEY, LANGFUSE_*, etc.
- For Chroma persistence, attach a Fly volume (see docs) and mount to /app/chroma_db in the backend service.
"
