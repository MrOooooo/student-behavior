#!/usr/bin/env bash
# Cloudflare Tunnel + Reflex 一键启动脚本
# 使用方法: bash start_tunnel.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLOUDFLARED="$HOME/.local/bin/cloudflared"
REFLEX="/home/server1/anaconda3/envs/eduview/bin/reflex"
CONDA_ENV="eduview"

export PATH="/home/server1/anaconda3/envs/eduview/bin:$HOME/.local/bin:$PATH"

echo "============================================"
echo "  学生行为智能检测 - 公网访问启动脚本"
echo "============================================"
echo ""

# 1. 清理旧进程
echo "[1/4] 清理旧进程..."
pkill -f "cloudflared tunnel" 2>/dev/null || true
pkill -f "reflex run" 2>/dev/null || true
pkill -f "gunicorn.*reflex" 2>/dev/null || true
pkill -f "next-server" 2>/dev/null || true
sleep 3

# 2. 启动后端隧道 (port 8000)
echo "[2/4] 启动后端隧道..."
$CLOUDFLARED tunnel --no-autoupdate --url http://localhost:8000 > /tmp/cf_backend.log 2>&1 &
BACKEND_PID=$!
sleep 6
BACKEND_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cf_backend.log | head -1)
echo "  后端: $BACKEND_URL"

if [ -z "$BACKEND_URL" ]; then
    echo "  ❌ 后端隧道启动失败！"
    cat /tmp/cf_backend.log
    exit 1
fi

# 3. 启动前端隧道 (port 3000)
echo "[3/4] 启动前端隧道..."
$CLOUDFLARED tunnel --no-autoupdate --url http://localhost:3000 > /tmp/cf_frontend.log 2>&1 &
FRONTEND_PID=$!
sleep 6
FRONTEND_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cf_frontend.log | head -1)
echo "  前端: $FRONTEND_URL"

if [ -z "$FRONTEND_URL" ]; then
    echo "  ❌ 前端隧道启动失败！"
    cat /tmp/cf_frontend.log
    exit 1
fi

# 4. 启动 Reflex
echo "[4/4] 启动 Reflex 应用..."
export API_URL="$BACKEND_URL"
export DEPLOY_URL="$FRONTEND_URL"
export EXTRA_CORS_ALLOWED_ORIGINS="$BACKEND_URL,$FRONTEND_URL,http://localhost:3001,http://127.0.0.1:3000"

cd "$SCRIPT_DIR"
$REFLEX run --env prod > /tmp/reflex.log 2>&1 &
REFLEX_PID=$!
sleep 30

# 验证
if curl -s --max-time 5 http://localhost:8000/ping | grep -q pong; then
    echo ""
    echo "============================================"
    echo "  ✅ 启动成功！"
    echo "  公网访问地址: $FRONTEND_URL"
    echo "============================================"
    echo ""
    echo "  进程 PID:"
    echo "    后端隧道: $BACKEND_PID"
    echo "    前端隧道: $FRONTEND_PID"
    echo "    Reflex:   $REFLEX_PID"
else
    echo "  ❌ Reflex 启动失败，查看日志: tail -f /tmp/reflex.log"
    exit 1
fi
