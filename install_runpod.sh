#!/bin/bash
# AI Toolkit RunPod 专用安装脚本
# 仓库: https://github.com/KeithZ117/ai-toolkit
# 优化用于 RunPod 环境

set -e

echo "=========================================="
echo "AI Toolkit - RunPod 安装"
echo "=========================================="
echo ""

# 检查 RunPod 环境
if [ -z "$RUNPOD_POD_ID" ]; then
    echo "⚠️  警告: 未检测到 RunPod 环境变量"
    echo "此脚本针对 RunPod 优化，但也可以在其他环境使用"
    echo ""
fi

# 使用 RunPod 标准路径
INSTALL_DIR="/workspace/ai-toolkit"
echo "Pod ID: ${RUNPOD_POD_ID:-N/A}"
echo "安装目录: $INSTALL_DIR"
echo ""

# 显示 GPU 信息
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo ""
fi

# ============================================
# 开始安装
# ============================================

echo "📦 步骤 1/7: 更新系统..."
apt-get update -qq

echo "🔧 步骤 2/7: 安装基础工具..."
apt-get install -y -qq git curl build-essential tmux htop unzip

echo "📦 步骤 3/7: 安装 Node.js..."
if ! command -v node &> /dev/null; then
    echo "   正在添加 NodeSource 仓库..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    echo "   正在安装 Node.js..."
    apt-get install -y nodejs
fi
echo "   ✓ Node.js $(node --version)"
echo "   ✓ npm $(npm --version)"

echo "📥 步骤 4/7: 克隆仓库..."
if [ -d "$INSTALL_DIR" ]; then
    echo "   更新现有仓库..."
    cd "$INSTALL_DIR"
    git pull origin main
else
    git clone https://github.com/KeithZ117/ai-toolkit.git "$INSTALL_DIR"
fi
cd "$INSTALL_DIR"

echo "🐍 步骤 5/7: 安装 Python 依赖..."
echo "   安装必要的 Python 包..."

# 安装 gdown
pip3 install --no-cache-dir gdown

# 安装项目依赖
echo "   安装项目依赖 (这可能需要几分钟)..."
pip3 install --no-cache-dir -r requirements.txt

echo "   ✓ Python 依赖安装完成"

echo "🎨 步骤 6/7: 安装和构建 UI..."
cd "$INSTALL_DIR/ui"
echo "   安装 npm 包..."
npm install
echo ""
echo "   构建 UI..."
npm run build
echo ""
echo "   初始化数据库..."
npm run update_db

echo "📁 步骤 7/7: 配置目录和权限..."
cd "$INSTALL_DIR"
mkdir -p datasets output output/tensorboard config
chmod -R 777 datasets output config

# 下载并解压数据集
echo ""
echo "📥 下载数据集..."
cd "$INSTALL_DIR/datasets"

# 从 Google Drive 下载
echo "   正在从 Google Drive 下载文件..."
gdown "https://drive.google.com/uc?id=1Rbc2J-erh8oQov608NesKQoLgktvcr1P" -O dataset.zip

if [ -f "dataset.zip" ]; then
    echo "   ✓ 下载完成"
    echo "   正在解压..."
    unzip dataset.zip
    echo "   ✓ 解压完成"
    
    # 删除压缩包（可选）
    rm dataset.zip
    echo "   ✓ 清理临时文件"
    
    # 显示下载的内容
    echo ""
    echo "📂 数据集内容:"
    ls -lh
else
    echo "   ❌ 下载失败"
fi

cd "$INSTALL_DIR"

# ============================================
# 验证安装
# ============================================

echo ""
echo "🔍 验证安装..."

# 验证 PyTorch
echo "   验证 Python 环境..."
python3 -c "
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"

# 验证数据库
if [ -f "$INSTALL_DIR/aitk_db.db" ]; then
    echo "✓ 数据库已初始化"
fi

# ============================================
# 创建快捷启动脚本
# ============================================

echo ""
echo "📝 创建快捷脚本..."

# UI 启动脚本
cat > "$INSTALL_DIR/start_ui.sh" << 'EOF'
#!/bin/bash
cd /workspace/ai-toolkit/ui
export HOST=0.0.0.0
export PORT=8675
npm run start
EOF
chmod +x "$INSTALL_DIR/start_ui.sh"

# TensorBoard 启动脚本
cat > "$INSTALL_DIR/start_tensorboard.sh" << 'EOF'
#!/bin/bash
tensorboard --logdir=/workspace/ai-toolkit/output/tensorboard --port=8188 --host=0.0.0.0
EOF
chmod +x "$INSTALL_DIR/start_tensorboard.sh"

# 显示访问信息脚本
cat > "$INSTALL_DIR/show_urls.sh" << 'EOFURLS'
#!/bin/bash
echo "=========================================="
echo "AI Toolkit 访问信息"
echo "=========================================="
echo ""
echo "Pod ID: $RUNPOD_POD_ID"
echo ""
echo "🎨 UI 界面:"
echo "  https://${RUNPOD_POD_ID}-8675.proxy.runpod.net"
echo ""
echo "📊 TensorBoard:"
echo "  https://${RUNPOD_POD_ID}-8188.proxy.runpod.net"
echo ""
echo "=========================================="
EOFURLS
chmod +x "$INSTALL_DIR/show_urls.sh"

# 一键启动所有服务
cat > "$INSTALL_DIR/start_all.sh" << 'EOFALL'
#!/bin/bash
echo "启动 UI..."
cd /workspace/ai-toolkit/ui
nohup npm run start > /tmp/ui.log 2>&1 &
echo "✓ UI 已在后台启动"

sleep 3

echo "启动 TensorBoard..."
nohup tensorboard --logdir=/workspace/ai-toolkit/output/tensorboard --port=8188 --host=0.0.0.0 > /tmp/tensorboard.log 2>&1 &
echo "✓ TensorBoard 已在后台启动"

sleep 2

bash /workspace/ai-toolkit/show_urls.sh
EOFALL
chmod +x "$INSTALL_DIR/start_all.sh"

# ============================================
# 完成
# ============================================

echo ""
echo "=========================================="
echo "✅ 安装完成！"
echo "=========================================="
echo ""
echo "📁 安装位置: $INSTALL_DIR"
echo ""
echo "🚀 快速启动命令:"
echo ""
echo "  # 一键启动所有服务"
echo "  bash $INSTALL_DIR/start_all.sh"
echo ""
echo "  # 或分别启动:"
echo "  bash $INSTALL_DIR/start_ui.sh"
echo "  bash $INSTALL_DIR/start_tensorboard.sh"
echo ""
echo "  # 查看访问链接"
echo "  bash $INSTALL_DIR/show_urls.sh"
echo ""
echo "🌐 访问地址:"

if [ -n "$RUNPOD_POD_ID" ]; then
    echo "  UI:          https://${RUNPOD_POD_ID}-8675.proxy.runpod.net"
    echo "  TensorBoard: https://${RUNPOD_POD_ID}-8188.proxy.runpod.net"
else
    echo "  UI:          http://localhost:8675"
    echo "  TensorBoard: http://localhost:8188"
fi

echo ""
echo "📖 示例配置:"
echo "  ls $INSTALL_DIR/config/examples/"
echo ""
echo "💡 重要提示:"
echo "  1. 设置 HuggingFace Token (用于 FLUX.1-dev):"
echo "     export HF_TOKEN='your_token_here'"
echo ""
echo "  2. 或创建 .env 文件:"
echo "     echo 'HF_TOKEN=your_token_here' > $INSTALL_DIR/.env"
echo ""
echo "=========================================="
echo "🎉 准备就绪！"
echo "=========================================="
