[CmdletBinding()]
param(
    [string]$Content = "00.png",
    [string]$Style = "00.png",
    [string]$SaveDir = "results\demo",

    [ValidateRange(1, 1000)]
    [int]$Steps = 20,

    [ValidateRange(64, 2048)]
    [int]$Resolution = 512,

    [ValidateRange(0.0, 1.0)]
    [double]$Gamma = 0.75,

    [ValidateRange(0.01, 20.0)]
    [double]$Temperature = 1.5,

    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",

    [ValidateSet("float16", "float32")]
    [string]$Precision = "float16",

    [int[]]$Layers = @(7, 8, 9, 10, 11),
    [switch]$WithoutInitAdaIN,
    [switch]$WithoutAttentionInjection,
    [switch]$SaveIntermediates,
    [switch]$Interactive,
    [switch]$VerifyWeights,
    [string]$ModelPath = ""
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Read-TextSetting {
    param([string]$Label, [string]$DefaultValue)
    $answer = Read-Host "$Label [$DefaultValue]"
    if ([string]::IsNullOrWhiteSpace($answer)) {
        return $DefaultValue
    }
    return $answer.Trim().Trim('"')
}

function Read-IntSetting {
    param([string]$Label, [int]$DefaultValue, [int]$Minimum, [int]$Maximum)
    while ($true) {
        $answer = Read-Host "$Label [$DefaultValue]"
        if ([string]::IsNullOrWhiteSpace($answer)) {
            return $DefaultValue
        }
        $parsed = 0
        if ([int]::TryParse($answer, [ref]$parsed) -and $parsed -ge $Minimum -and $parsed -le $Maximum) {
            return $parsed
        }
        Write-Host "Please enter an integer from $Minimum to $Maximum." -ForegroundColor Yellow
    }
}

function Read-DoubleSetting {
    param([string]$Label, [double]$DefaultValue, [double]$Minimum, [double]$Maximum)
    while ($true) {
        $answer = Read-Host "$Label [$DefaultValue]"
        if ([string]::IsNullOrWhiteSpace($answer)) {
            return $DefaultValue
        }
        $parsed = 0.0
        if ([double]::TryParse($answer, [ref]$parsed) -and $parsed -ge $Minimum -and $parsed -le $Maximum) {
            return $parsed
        }
        Write-Host "Please enter a number from $Minimum to $Maximum." -ForegroundColor Yellow
    }
}

function Resolve-ImagePath {
    param([string]$Value, [string]$DefaultDirectory)
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return $Value
    }
    if ($Value.Contains('\') -or $Value.Contains('/')) {
        return Join-Path $PSScriptRoot $Value
    }
    return Join-Path (Join-Path $PSScriptRoot $DefaultDirectory) $Value
}

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $PSScriptRoot ".cache\huggingface\hub\models--stable-diffusion-v1-5--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
}

if ($Interactive) {
    Write-Host "StyleID interactive launcher" -ForegroundColor Cyan
    Write-Host "Press Enter to keep the value shown in brackets."
    Write-Host ""

    $Content = Read-TextSetting "Content image name (from data\cnt)" ([System.IO.Path]::GetFileName($Content))
    $Style = Read-TextSetting "Style image name (from data\sty)" ([System.IO.Path]::GetFileName($Style))
    $Steps = Read-IntSetting "DDIM steps" $Steps 1 1000
    $Resolution = Read-IntSetting "Resolution" $Resolution 64 2048
    $Gamma = Read-DoubleSetting "Gamma" $Gamma 0.0 1.0
    $Temperature = Read-DoubleSetting "Temperature (T)" $Temperature 0.01 20.0

    Write-Host ""
}

$Content = Resolve-ImagePath $Content "data\cnt"
$Style = Resolve-ImagePath $Style "data\sty"
$contentStem = [System.IO.Path]::GetFileNameWithoutExtension($Content)
$styleStem = [System.IO.Path]::GetFileNameWithoutExtension($Style)
$gammaTag = $Gamma.ToString("0.####", [System.Globalization.CultureInfo]::InvariantCulture).Replace('.', 'p')
$temperatureTag = $Temperature.ToString("0.####", [System.Globalization.CultureInfo]::InvariantCulture).Replace('.', 'p')
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$outputName = "c${contentStem}_s${styleStem}_d${Steps}_r${Resolution}_g${gammaTag}_t${temperatureTag}_${runStamp}.jpg"

if ($Resolution % 8 -ne 0) {
    throw "Resolution must be divisible by 8. Received: $Resolution"
}

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$runnerPath = Join-Path $PSScriptRoot "diffusers_implementation\run_styleid_diffusers.py"

foreach ($requiredPath in @($pythonPath, $runnerPath, $Content, $Style, $ModelPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

$weightFiles = [ordered]@{
    "unet\diffusion_pytorch_model.fp16.safetensors" = "C83908253F9A64D08C25FC90874C9C8AEF9A329CE1CA5FB909D73B0C83D1EA21"
    "text_encoder\model.fp16.safetensors" = "77795E2023ADCF39BC29A884661950380BD093CF0750A966D473D1718DC9EF4E"
    "vae\diffusion_pytorch_model.fp16.safetensors" = "4FBCF0EBE55A0984F5A5E00D8C4521D52359AF7229BB4D81890039D2AA16DD7C"
}

foreach ($relativeWeightPath in $weightFiles.Keys) {
    $weightPath = Join-Path $ModelPath $relativeWeightPath
    if (-not (Test-Path -LiteralPath $weightPath)) {
        throw "Model weight is missing: $weightPath"
    }
    if ($VerifyWeights) {
        Write-Host "Verifying $relativeWeightPath ..."
        $actualHash = (Get-FileHash -LiteralPath $weightPath -Algorithm SHA256).Hash
        if ($actualHash -ne $weightFiles[$relativeWeightPath]) {
            throw "SHA-256 mismatch: $weightPath"
        }
    }
}

if ($Device -eq "cpu" -and $Precision -eq "float16") {
    Write-Warning "CPU mode does not reliably support float16; using float32."
    $Precision = "float32"
}

Write-Host "StyleID settings" -ForegroundColor Cyan
Write-Host "  Content:    $Content"
Write-Host "  Style:      $Style"
Write-Host "  Output:     $SaveDir"
Write-Host "  Steps:      $Steps"
Write-Host "  Resolution: $Resolution"
Write-Host "  Gamma:      $Gamma"
Write-Host "  T:          $Temperature"
Write-Host "  Device:     $Device"
Write-Host "  Precision:  $Precision"
Write-Host "  Layers:     $($Layers -join ',')"
Write-Host ""

if ($Interactive) {
    $confirmation = Read-Host "Press Enter to start, or type Q to cancel"
    if ($confirmation -match '^[Qq]$') {
        Write-Host "Cancelled."
        exit 0
    }
}

$pythonArguments = @(
    $runnerPath,
    "--cnt_fn", $Content,
    "--sty_fn", $Style,
    "--model_id", $ModelPath,
    "--local_files_only",
    "--device", $Device,
    "--precision", $Precision,
    "--resolution", $Resolution,
    "--ddim_steps", $Steps,
    "--gamma", $Gamma,
    "--T", $Temperature,
    "--save_dir", $SaveDir,
    "--output_name", $outputName,
    "--layers"
)
$pythonArguments += $Layers

if ($WithoutInitAdaIN) {
    $pythonArguments += "--without_init_adain"
}
if ($WithoutAttentionInjection) {
    $pythonArguments += "--without_attn_injection"
}
if ($SaveIntermediates) {
    $pythonArguments += "--save_intermediates"
}

$env:HF_HOME = Join-Path $PSScriptRoot ".cache\huggingface"
$env:HF_HUB_OFFLINE = "1"

& $pythonPath @pythonArguments
if ($LASTEXITCODE -ne 0) {
    throw "StyleID exited with code $LASTEXITCODE."
}

$resultPath = Join-Path (Join-Path $PSScriptRoot $SaveDir) $outputName
Write-Host ""
Write-Host "Completed: $resultPath" -ForegroundColor Green

