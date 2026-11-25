$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# CUDA 13.0 wheels support the RTX 50 series and include their own CUDA runtime.
& $python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
& $python -m pip install -r requirements-diffusers.txt

$env:HF_HOME = Join-Path $PSScriptRoot ".cache\huggingface"
if ([string]::IsNullOrWhiteSpace($env:HF_ENDPOINT)) {
    $env:HF_ENDPOINT = "https://hf-mirror.com"
}
$env:HF_HUB_DOWNLOAD_TIMEOUT = "300"
$env:HF_HUB_ETAG_TIMEOUT = "60"
$env:HF_HUB_DISABLE_XET = "1"

$hf = Join-Path $PSScriptRoot ".venv\Scripts\hf.exe"
$modelFiles = @(
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors"
)
& $hf download stable-diffusion-v1-5/stable-diffusion-v1-5 @modelFiles `
    --revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14

& $python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
