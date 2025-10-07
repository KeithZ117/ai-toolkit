#!/bin/bash
# AI Toolkit 快速安装脚本 (简化版)
# 仓库: https://github.com/KeithZ117/ai-toolkit

set -e

echo "=========================================="
echo "AI Toolkit 快速安装"
echo "=========================================="
echo ""

# 安装目录
INSTALL_DIR="${1:-/workspace/ai-toolkit}"
echo "安装到: $INSTALL_DIR"
echo ""

# 1. 更新系统
echo "📦 更新系统包..."
apt-get update -qq

# 2. 安装基础工具
echo "🔧 安装基础工具..."
apt-get install -y -qq git curl build-essential

# 3. 安装 Node.js
echo "📦 安装 Node.js..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
echo "   Node.js: $(node --version)"
echo "   npm: $(npm --version)"

# 4. 克隆仓库
echo "📥 克隆仓库..."
if [ -d "$INSTALL_DIR" ]; then
    echo "   目录已存在，更新中..."
    cd "$INSTALL_DIR" && git pull
else
    git clone https://github.com/KeithZ117/ai-toolkit.git "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# 5. 安装 Python 依赖
echo "🐍 安装 Python 依赖..."
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install --no-cache-dir -r requirements.txt

# 6. 安装 UI 依赖
echo "🎨 安装 UI 依赖..."
cd ui
npm install --silent
npm run build
npm run update_db

# 7. 创建目录
echo "📁 创建必要目录..."
cd "$INSTALL_DIR"
mkdir -p datasets output output/.tensorboard config
chmod -R 777 datasets output config

echo ""
echo "=========================================="
echo "✅ 安装完成！"
echo "=========================================="
echo ""
echo "🚀 启动 UI:"
echo "  cd $INSTALL_DIR/ui && npm run start"
echo ""
echo "📊 启动 TensorBoard:"
echo "  tensorboard --logdir=$INSTALL_DIR/output/.tensorboard --bind_all"
echo ""
