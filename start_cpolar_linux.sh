#!/usr/bin/env bash
set -euo pipefail

# Usage: bash start_cpolar_linux.sh <public_backend_url> <public_frontend_url>
# Example: bash start_cpolar_linux.sh https://6a5ac73a.r9.vip.cpolar.cn https://23369c32.r26.cpolar.top

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <public_backend_url> <public_frontend_url>"
  echo "Backend URL maps to local port 8000; frontend URL maps to local port 3000unset CORS_ALLOWED_ORIGINS
rm -f .web/env.json."
  exit 1
fi

export API_URL="$1"
export DEPLOY_URL="$2"
export EXTRA_CORS_ALLOWED_ORIGINS="$1,$2,http://localhost:3001,http://127.0.0.1:3000"

cd "$(dirname "$0")"

echo "API_URL=$API_URL"
echo "DEPLOY_URL=$DEPLOY_URL"
echo "EXTRA_CORS_ALLOWED_ORIGINS=$EXTRA_CORS_ALLOWED_ORIGINS"
echo "Starting Reflex: frontend=3000 backend=8000"

reflex run --env prod
