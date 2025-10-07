#!/bin/bash
# AI Toolkit RunPod 精简安装脚本
# 仓库: https://github.com/KeithZ117/ai-toolkit
# 假设 Python 环境已配置好

set -e

echo "=========================================="
echo "AI Toolkit - 快速部署 (RunPod)"
echo "=========================================="
echo ""

INSTALL_DIR="/workspace/ai-toolkit"

# 显示环境信息
echo "Pod ID: ${RUNPOD_POD_ID:-N/A}"
echo "安装目录: $INSTALL_DIR"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo ""
fi

# ============================================
# 安装步骤
# ============================================

echo "📦 步骤 1/4: 安装基础工具..."
apt-get update -qq
apt-get install -y -qq git curl tmux htop

echo "📦 步骤 2/4: 安装 Node.js..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
    apt-get install -y nodejs > /dev/null 2>&1
fi
echo "   ✓ Node.js $(node --version)"

echo "📥 步骤 3/4: 获取代码..."
if [ -d "$INSTALL_DIR" ]; then
    cd "$INSTALL_DIR" && git pull origin main
else
    git clone https://github.com/KeithZ117/ai-toolkit.git "$INSTALL_DIR"
fi

echo "🎨 步骤 4/4: 构建 UI..."
cd "$INSTALL_DIR/ui"
npm install --silent
npm run build
npm run update_db

# 创建目录
cd "$INSTALL_DIR"
mkdir -p datasets output output/.tensorboard config
chmod -R 777 datasets output config

# ============================================
# 创建启动脚本
# ============================================

cat > "$INSTALL_DIR/start_all.sh" << 'EOF'
#!/bin/bash
cd /workspace/ai-toolkit/ui
nohup npm run start > /tmp/ui.log 2>&1 &
echo "✓ UI 启动: https://${RUNPOD_POD_ID}-8675.proxy.runpod.net"

nohup tensorboard --logdir=/workspace/ai-toolkit/output/.tensorboard --port=6006 --host=0.0.0.0 > /tmp/tensorboard.log 2>&1 &
echo "✓ TensorBoard 启动: https://${RUNPOD_POD_ID}-6006.proxy.runpod.net"
EOF
chmod +x "$INSTALL_DIR/start_all.sh"

# ============================================
# 完成
# ============================================

echo ""
echo "=========================================="
echo "✅ 安装完成！"
echo "=========================================="
echo ""
echo "🚀 启动服务:"
echo "  bash $INSTALL_DIR/start_all.sh"
echo ""
echo "🌐 访问地址:"
echo "  UI:          https://${RUNPOD_POD_ID:-localhost}-8675.proxy.runpod.net"
echo "  TensorBoard: https://${RUNPOD_POD_ID:-localhost}-6006.proxy.runpod.net"
echo ""
