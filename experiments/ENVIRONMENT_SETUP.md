# Linux 服务器环境配置指南

适用场景：服务器 NVIDIA 驱动对应 CUDA 12.8，实测 Python 3.12.3 + CUDA 12.4 版 PyTorch 组合稳定可用。国内服务器统一使用阿里云镜像加速。

---

## 1. 配置阿里云镜像源

### pip 源

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

### conda 源

```bash
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main/
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

---

## 2. 创建 conda 环境

```bash
conda create -n deep python=3.12.3 -y
conda activate deep
```

## 3. 安装 PyTorch（CUDA 12.4，阿里云镜像）

> 注意：`https://mirrors.aliyun.com/pytorch-wheels/cu124/` 是一个**平铺的文件目录**，不是标准 PEP503 索引，不能用 `--index-url`，要用 `--find-links`（否则会报 `No matching distribution found for torch`）。

```bash
pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 torchaudio==2.5.0+cu124 --find-links https://mirrors.aliyun.com/pytorch-wheels/cu124/
```

> 服务器驱动是 CUDA 12.8 没关系，PyTorch 的 cu124 wheel 向下兼容，驱动版本只要 ≥ 12.4 即可。其余纯 Python 依赖（sympy/jinja2/nvidia-*-cu12 等）会自动从第 1 步配置好的阿里云 pip 源下载。

## 4. 安装其余依赖

```bash
cd ~/SpectrumAnalyzer/experiments
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
numpy
pandas
pillow
scikit-learn
tqdm
```

## 5. 验证安装

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

预期输出类似：

```
2.x.x+cu124 True
```

没有报错、`cuda.is_available()` 为 `True` 即为安装成功。

---

## 6. 运行实验

```bash
# 单个 backbone 调试
python train.py --backbone resnet50 --epochs 50

# 一键跑全部 backbone
bash run_all.sh

# 汇总结果
python summarize_results.py
```

---

## 常见问题

### `ImportError: ... libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`

**原因**：conda 环境里的 `mkl` 版本过新（2024.1/2024.2 起移除了该符号），与 PyTorch 依赖的旧版 MKL 接口不兼容。

**排查**：

```bash
conda list mkl
```

**修复**：

```bash
conda install "mkl=2021.4.0" -c conda-forge --no-update-deps
```

若仍报错，彻底重装 PyTorch：

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 torchaudio==2.5.0+cu124 --find-links https://mirrors.aliyun.com/pytorch-wheels/cu124/
```
