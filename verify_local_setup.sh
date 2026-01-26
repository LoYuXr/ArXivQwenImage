#!/bin/bash

# =========================================
# 本地配置验证脚本
# =========================================
# 在运行训练前，验证所有路径和环境配置
# =========================================

echo "=========================================="
echo "Flux2Klein 本地配置验证"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查计数
PASS=0
FAIL=0

# ====== 1. 检查 Conda 环境 ======
echo "[1/8] 检查 Conda 环境..."
if conda info --envs | grep -q "flux2"; then
    echo -e "${GREEN}✓${NC} Conda 环境 'flux2' 已安装"
    PASS=$((PASS+1))
else
    echo -e "${RED}✗${NC} Conda 环境 'flux2' 未找到"
    echo "  请运行: conda create -n flux2 python=3.10"
    FAIL=$((FAIL+1))
fi
echo ""

# ====== 2. 检查 GPU ======
echo "[2/8] 检查 GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo -e "${GREEN}✓${NC} 检测到 $GPU_COUNT 个 GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | nl -v 0
    PASS=$((PASS+1))
else
    echo -e "${RED}✗${NC} 未检测到 NVIDIA GPU 或 nvidia-smi 未安装"
    FAIL=$((FAIL+1))
fi
echo ""

# ====== 3. 检查数据集路径（本地） ======
echo "[3/8] 检查数据集路径（本地）..."
DATASET_PATH="/home/v-yuxluo/yuxuanluo/ArXiV_parquet/Flux2Klein9BParquet_260118"
if [ -d "$DATASET_PATH" ]; then
    echo -e "${GREEN}✓${NC} 数据集路径存在"
    echo "  路径: $DATASET_PATH"
    
    # 检查是否有年份目录
    YEAR_COUNT=$(ls -1 "$DATASET_PATH" | wc -l)
    echo "  包含 $YEAR_COUNT 个目录"
    
    # 检查是否有 parquet 文件
    PARQUET_COUNT=$(find "$DATASET_PATH" -name "*.parquet" -type f | wc -l)
    echo "  包含 $PARQUET_COUNT 个 parquet 文件"
    
    if [ $PARQUET_COUNT -gt 0 ]; then
        echo -e "${GREEN}✓${NC} 数据集包含有效的 parquet 文件"
        PASS=$((PASS+1))
    else
        echo -e "${YELLOW}⚠${NC} 数据集目录存在但未找到 parquet 文件"
        FAIL=$((FAIL+1))
    fi
else
    echo -e "${RED}✗${NC} 数据集路径不存在: $DATASET_PATH"
    echo "  期望的本地路径: /home/v-yuxluo/yuxuanluo/ArXiV_parquet/Flux2Klein9BParquet_260118"
    echo "  (等价于远程 AMLT 的 /mnt/data/ArXiV_parquet/Flux2Klein9BParquet_260118)"
    FAIL=$((FAIL+1))
fi
echo ""

# ====== 4. 检查输出目录权限 ======
echo "[4/8] 检查输出目录..."
OUTPUT_DIR="/home/v-yuxluo/WORK_local/ArXivQwenImage/output/flux2klein_fulltune_debug"
mkdir -p "$OUTPUT_DIR" 2>/dev/null
if [ -w "$OUTPUT_DIR" ]; then
    echo -e "${GREEN}✓${NC} 输出目录可写"
    echo "  路径: $OUTPUT_DIR"
    PASS=$((PASS+1))
else
    echo -e "${RED}✗${NC} 输出目录不可写: $OUTPUT_DIR"
    FAIL=$((FAIL+1))
fi
echo ""

# ====== 5. 检查配置文件 ======
echo "[5/8] 检查配置文件..."
CONFIG_FILE="/home/v-yuxluo/WORK_local/ArXivQwenImage/configs/260121/flux2klein_fulltune_local_debug.py"
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${GREEN}✓${NC} 配置文件存在"
    echo "  路径: $CONFIG_FILE"
    
    # 检查是否使用本地路径
    if grep -q "/home/v-yuxluo/yuxuanluo" "$CONFIG_FILE"; then
        echo -e "${GREEN}✓${NC} 配置文件使用本地路径"
        PASS=$((PASS+1))
    else
        echo -e "${YELLOW}⚠${NC} 配置文件可能使用了远程路径 /mnt/data"
        echo "  请确认 base_dir 设置为: /home/v-yuxluo/yuxuanluo"
        FAIL=$((FAIL+1))
    fi
else
    echo -e "${RED}✗${NC} 配置文件不存在: $CONFIG_FILE"
    FAIL=$((FAIL+1))
fi
echo ""

# ====== 6. 检查训练脚本 ======
echo "[6/8] 检查训练脚本..."
TRAIN_SCRIPT="/home/v-yuxluo/WORK_local/ArXivQwenImage/train_OpenSciDraw_fulltune.py"
if [ -f "$TRAIN_SCRIPT" ]; then
    echo -e "${GREEN}✓${NC} 训练脚本存在"
    echo "  路径: $TRAIN_SCRIPT"
    PASS=$((PASS+1))
else
    echo -e "${RED}✗${NC} 训练脚本不存在: $TRAIN_SCRIPT"
    FAIL=$((FAIL+1))
fi
echo ""

# ====== 7. 检查 Accelerate 配置 ======
echo "[7/8] 检查 Accelerate 配置..."
ACCELERATE_CFG="/home/v-yuxluo/WORK_local/ArXivQwenImage/accelerate_cfg/1m4g_bf16.yaml"
if [ -f "$ACCELERATE_CFG" ]; then
    echo -e "${GREEN}✓${NC} Accelerate 配置文件存在"
    echo "  路径: $ACCELERATE_CFG"
    PASS=$((PASS+1))
else
    echo -e "${YELLOW}⚠${NC} Accelerate 配置文件不存在: $ACCELERATE_CFG"
    echo "  将使用默认配置"
    FAIL=$((FAIL+1))
fi
echo ""

# ====== 8. 检查 HuggingFace Token ======
echo "[8/8] 检查 HuggingFace Token..."
if [ -n "$HF_TOKEN" ]; then
    echo -e "${GREEN}✓${NC} HF_TOKEN 环境变量已设置"
    PASS=$((PASS+1))
else
    HF_TOKEN_FILE="$HOME/.huggingface/token"
    if [ -f "$HF_TOKEN_FILE" ]; then
        echo -e "${GREEN}✓${NC} HuggingFace token 文件存在"
        PASS=$((PASS+1))
    else
        echo -e "${YELLOW}⚠${NC} HuggingFace token 未配置"
        echo "  建议设置环境变量: export HF_TOKEN=your_token"
        echo "  或运行: huggingface-cli login"
    fi
fi
echo ""

# ====== 总结 ======
echo "=========================================="
echo "验证结果总结"
echo "=========================================="
echo -e "通过: ${GREEN}$PASS${NC}"
echo -e "失败: ${RED}$FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ 所有检查通过！可以开始训练。${NC}"
    echo ""
    echo "运行训练命令:"
    echo "  bash run_fulltune_local.sh"
    exit 0
else
    echo -e "${RED}✗ 存在 $FAIL 个问题，请先修复后再继续。${NC}"
    exit 1
fi
