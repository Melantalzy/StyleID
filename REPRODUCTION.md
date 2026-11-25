# StyleID 本地复现说明（Windows / RTX 5060 Ti 8GB）

本工作区保留了上游原始实现，并对 `diffusers_implementation` 做了 Windows、当前 PyTorch 与 8GB 显存兼容。默认使用公开可直接下载的 Stable Diffusion 1.5 Diffusers 权重。

## 已验证环境

- Windows，NVIDIA GeForce RTX 5060 Ti 8GB
- Python 3.14.6
- PyTorch 2.13.0 + CUDA 13.0
- torchvision 0.28.0
- diffusers 0.41.0
- transformers 5.20.1

## 一键安装

在 PowerShell 中运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

安装脚本会创建项目内的 `.venv`，安装 RTX 50 系可用的 CUDA 版 PyTorch，
通过 `hf download` 下载运行所需的 Stable Diffusion 1.5 fp16 文件，并做 CUDA 检查。
默认使用 `https://hf-mirror.com`；如果运行前已经设置 `HF_ENDPOINT`，则保留你的配置。

## 运行样例

```powershell
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
```

结果保存到 `results/demo/`。权重由安装脚本下载到 `.cache/huggingface`，后续运行复用缓存。

也可以指定输入和参数：

```powershell
.\run_demo.ps1 `
  -Content data\cnt\01.png `
  -Style data\sty\08.png `
  -SaveDir results\custom `
  -Steps 20 `
  -Resolution 512 `
  -Gamma 0.75 `
  -Temperature 1.5
```

`Gamma` 越高越忠于内容，越低越忠于风格；`Temperature` 控制注意力温度。自定义图片会先中心裁剪，再缩放为相同的正方形分辨率。

## 显存与论文复现边界

上游 README 所述的原始 LDM + SD 1.4 路径需要 20GB 以上显存。这里的可运行路径采用仓库官方提供的 Diffusers 实现，并做了等价的内存优化：注意力注入使用 PyTorch SDPA，风格反演只保存 K/V，内容反演只保存 Q。核心的 query preservation、style K/V injection 与 latent AdaIN 逻辑均保留。

如需逐步反演图和去噪图，在 Python 命令末尾增加 `--save_intermediates`；它会显著增加运行时间和磁盘占用。

离线复用已下载模型时可在 Python 命令中增加 `--local_files_only`。
