#!/bin/bash
# AI Toolkit 安装脚本
# 使用 KeithZ117 的 fork 仓库
# 适用于 Ubuntu/Debian 系统和 RunPod 环境

set -e  # 遇到错误立即退出

echo "=========================================="
echo "AI Toolkit 安装脚本"
echo "仓库: https://github.com/KeithZ117/ai-toolkit"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检测系统
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
    echo "检测到系统: $OS $VER"
else
    echo "无法检测系统类型"
    OS="Unknown"
fi
echo ""

# 检查是否在 RunPod 环境
if [ -n "$RUNPOD_POD_ID" ]; then
    echo -e "${GREEN}✓${NC} 检测到 RunPod 环境 (Pod ID: $RUNPOD_POD_ID)"
    IN_RUNPOD=true
else
    echo "在本地或其他云环境中运行"
    IN_RUNPOD=false
fi
echo ""

# 设置安装目录
if [ -d "/workspace" ]; then
    INSTALL_DIR="/workspace/ai-toolkit"
    echo "使用安装目录: $INSTALL_DIR (RunPod 标准路径)"
elif [ -d "/app" ]; then
    INSTALL_DIR="/app/ai-toolkit"
    echo "使用安装目录: $INSTALL_DIR"
else
    INSTALL_DIR="$HOME/ai-toolkit"
    echo "使用安装目录: $INSTALL_DIR"
fi
echo ""

# 检查是否以 root 运行
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}⚠${NC} 未以 root 用户运行，某些操作可能需要 sudo"
    SUDO="sudo"
else
    echo -e "${GREEN}✓${NC} 以 root 用户运行"
    SUDO=""
fi
echo ""

# ============================================
# 步骤 1: 更新系统包
# ============================================
echo "=========================================="
echo "步骤 1/8: 更新系统包"
echo "=========================================="
$SUDO apt-get update -qq
echo -e "${GREEN}✓${NC} 系统包已更新"
echo ""

# ============================================
# 步骤 2: 安装基础工具
# ============================================
echo "=========================================="
echo "步骤 2/8: 安装基础工具"
echo "=========================================="
$SUDO apt-get install -y -qq \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    ffmpeg \
    tmux \
    htop \
    > /dev/null 2>&1
echo -e "${GREEN}✓${NC} 基础工具安装完成"
echo ""

# ============================================
# 步骤 3: 检查 Python
# ============================================
echo "=========================================="
echo "步骤 3/8: 检查 Python 环境"
echo "=========================================="
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Python 已安装: $PYTHON_VERSION"
else
    echo "安装 Python 3..."
    $SUDO apt-get install -y python3 python3-pip python3-venv > /dev/null 2>&1
    echo -e "${GREEN}✓${NC} Python 安装完成"
fi

# 检查 pip
if command -v pip3 &> /dev/null; then
    PIP_VERSION=$(pip3 --version | awk '{print $2}')
    echo -e "${GREEN}✓${NC} pip 已安装: $PIP_VERSION"
else
    echo "安装 pip..."
    $SUDO apt-get install -y python3-pip > /dev/null 2>&1
    echo -e "${GREEN}✓${NC} pip 安装完成"
fi
echo ""

# ============================================
# 步骤 4: 安装 Node.js 和 npm
# ============================================
echo "=========================================="
echo "步骤 4/8: 安装 Node.js 和 npm"
echo "=========================================="
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓${NC} Node.js 已安装: $NODE_VERSION"
else
    echo "安装 Node.js 20.x..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | $SUDO bash - > /dev/null 2>&1
    $SUDO apt-get install -y nodejs > /dev/null 2>&1
    echo -e "${GREEN}✓${NC} Node.js 安装完成: $(node --version)"
fi

if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✓${NC} npm 已安装: $NPM_VERSION"
fi
echo ""

# ============================================
# 步骤 5: 克隆或更新仓库
# ============================================
echo "=========================================="
echo "步骤 5/8: 获取 AI Toolkit 代码"
echo "=========================================="

if [ -d "$INSTALL_DIR" ]; then
    echo "目录已存在: $INSTALL_DIR"
    read -p "是否删除并重新克隆? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除旧目录..."
        rm -rf "$INSTALL_DIR"
        echo "克隆仓库..."
        git clone https://github.com/KeithZ117/ai-toolkit.git "$INSTALL_DIR"
        echo -e "${GREEN}✓${NC} 仓库克隆完成"
    else
        echo "更新现有仓库..."
        cd "$INSTALL_DIR"
        git pull origin main
        echo -e "${GREEN}✓${NC} 仓库更新完成"
    fi
else
    echo "克隆仓库到: $INSTALL_DIR"
    git clone https://github.com/KeithZ117/ai-toolkit.git "$INSTALL_DIR"
    echo -e "${GREEN}✓${NC} 仓库克隆完成"
fi

cd "$INSTALL_DIR"
echo ""

# ============================================
# 步骤 6: 安装 Python 依赖
# ============================================
echo "=========================================="
echo "步骤 6/8: 安装 Python 依赖"
echo "=========================================="

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} 检测到 NVIDIA GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    CUDA_AVAILABLE=true
else
    echo -e "${YELLOW}⚠${NC} 未检测到 NVIDIA GPU"
    CUDA_AVAILABLE=false
fi
echo ""

# 安装 PyTorch
echo "安装 PyTorch (CUDA 12.6)..."
pip3 install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126
echo -e "${GREEN}✓${NC} PyTorch 安装完成"
echo ""

# 安装其他依赖
echo "安装其他 Python 依赖..."
pip3 install --no-cache-dir -r requirements.txt
echo -e "${GREEN}✓${NC} Python 依赖安装完成"
echo ""

# ============================================
# 步骤 7: 安装 UI 依赖
# ============================================
echo "=========================================="
echo "步骤 7/8: 安装 UI 依赖"
echo "=========================================="

cd "$INSTALL_DIR/ui"

# 安装 npm 包
echo "安装 npm 包..."
npm install --silent
echo -e "${GREEN}✓${NC} npm 包安装完成"
echo ""

# 构建 UI
echo "构建 UI..."
npm run build
echo -e "${GREEN}✓${NC} UI 构建完成"
echo ""

# 初始化数据库
echo "初始化数据库..."
npm run update_db
echo -e "${GREEN}✓${NC} 数据库初始化完成"
echo ""

# ============================================
# 步骤 8: 创建必要的目录
# ============================================
echo "=========================================="
echo "步骤 8/8: 创建必要的目录"
echo "=========================================="

cd "$INSTALL_DIR"
mkdir -p datasets
mkdir -p output
mkdir -p output/.tensorboard
mkdir -p config

# 设置权限
chmod -R 755 "$INSTALL_DIR"
chmod -R 777 "$INSTALL_DIR/datasets"
chmod -R 777 "$INSTALL_DIR/output"
chmod -R 777 "$INSTALL_DIR/config"

echo -e "${GREEN}✓${NC} 目录创建完成"
echo ""

# ============================================
# 验证安装
# ============================================
echo "=========================================="
echo "验证安装"
echo "=========================================="

cd "$INSTALL_DIR"

# 验证 Python 导入
echo "验证 Python 环境..."
python3 -c "
import torch
import diffusers
import transformers
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
print(f'✓ Diffusers: {diffusers.__version__}')
print(f'✓ Transformers: {transformers.__version__}')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Python 环境验证通过"
else
    echo -e "${RED}✗${NC} Python 环境验证失败"
fi
echo ""

# 验证 Node.js
echo "验证 Node.js 环境..."
echo "  Node.js: $(node --version)"
echo "  npm: $(npm --version)"
echo -e "${GREEN}✓${NC} Node.js 环境验证通过"
echo ""

# 验证数据库
if [ -f "$INSTALL_DIR/aitk_db.db" ]; then
    echo -e "${GREEN}✓${NC} 数据库文件已创建"
else
    echo -e "${YELLOW}⚠${NC} 数据库文件未找到"
fi
echo ""

# ============================================
# 安装完成
# ============================================
echo "=========================================="
echo -e "${GREEN}✓ 安装完成！${NC}"
echo "=========================================="
echo ""
echo "📁 安装位置: $INSTALL_DIR"
echo ""
echo "🚀 启动 UI:"
echo "  cd $INSTALL_DIR/ui"
echo "  npm run start"
echo ""
echo "  访问: http://localhost:8675"

if [ "$IN_RUNPOD" = true ]; then
    echo "  或: https://${RUNPOD_POD_ID}-8675.proxy.runpod.net"
fi

echo ""
echo "📊 启动 TensorBoard:"
echo "  tensorboard --logdir=$INSTALL_DIR/output/.tensorboard --port=6006 --host=0.0.0.0"

if [ "$IN_RUNPOD" = true ]; then
    echo "  访问: https://${RUNPOD_POD_ID}-6006.proxy.runpod.net"
fi

echo ""
echo "🎓 开始训练:"
echo "  cd $INSTALL_DIR"
echo "  python run.py config/your_config.yaml"
echo ""
echo "📖 查看示例配置:"
echo "  ls $INSTALL_DIR/config/examples/"
echo ""
echo "💡 提示:"
echo "  - 训练前需要设置 HuggingFace token (如果使用 FLUX.1-dev):"
echo "    export HF_TOKEN='your_token_here'"
echo "  - 或创建 .env 文件:"
echo "    echo 'HF_TOKEN=your_token_here' > $INSTALL_DIR/.env"
echo ""
echo "=========================================="
echo -e "${GREEN}祝训练顺利！🎉${NC}"
echo "=========================================="
