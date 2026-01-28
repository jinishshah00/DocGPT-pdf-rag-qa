REMOVED: deployment guidance moved to chat by user request
This file is disabled to avoid leaving guidance in the repo.
Please refer to the chat for the latest instructions.
All previous content has been removed.

Prerequisites
- Install Docker and confirm you can build images locally.
- Install `flyctl` (https://fly.io/docs/hands-on/install-flyctl/).
- Create a Fly account and run `flyctl auth login`.

High-level steps
1. Build and test locally with Docker Compose (optional):

   docker compose -f infra/docker-compose.yml up --build

2. Create backend app on Fly and a managed Postgres:

   flyctl launch --name docgpt-backend --no-deploy
   flyctl postgres create --name docgpt-db --region <region>

3. Attach Postgres to backend and set secrets (get the DATABASE_URL from Fly):

   # set secrets (example)
   flyctl secrets set OPENAI_API_KEY=sk-... JWT_SECRET=change-me DATABASE_URL=postgres://... TAVILY_API_KEY=... LANGFUSE_SECRET_KEY=...

4. Deploy backend

   # from repo root, ensure infra/backend.Dockerfile is referenced in fly.toml or use build args
   flyctl deploy --app docgpt-backend --config deploy/fly_backend.toml.template

5. Create frontend app and deploy

   flyctl launch --name docgpt-frontend --no-deploy
   # set frontend secret pointing to backend public URL
   flyctl secrets set API_BASE_PUBLIC=https://<docgpt-backend>.fly.dev
   flyctl deploy --app docgpt-frontend --config deploy/fly_frontend.toml.template

6. Mount persistent volume for Chroma

   flyctl volumes create chroma_volume --region <region> --size 1
   # attach in fly.toml using the mounts section

DNS / custom domain (optional)
- Add a CNAME or A record pointing to the Fly-provided hostname. See Fly docs for domain setup.

Notes & caveats
- Streamlit apps run on port 8501 by default; Fly's service port mapping will expose via port 80/443.
- Chroma requires a writable filesystem for its sqlite files. Use Fly volumes for persistence and mount to `/app/chroma_db` in the backend container.
- For small demo usage, Fly free allowances are typically sufficient; check quotas and billing before production use.

If you want, I can:
- Create finalized `fly.toml` files in `backend/` and `frontend/` directories (requires you to confirm app names and regions), or
- Walk you through running the exact flyctl commands interactively in your terminal and explain any errors.
