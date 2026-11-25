param(
    [string]$Content = "data\cnt\00.png",
    [string]$Style = "data\sty\00.png",
    [string]$SaveDir = "results\demo",
    [int]$Steps = 20,
    [int]$Resolution = 512,
    [double]$Gamma = 0.75,
    [double]$Temperature = 1.5
)

$ErrorActionPreference = "Stop"

& "$PSScriptRoot\run_styleid.ps1" `
    -Content $Content `
    -Style $Style `
    -SaveDir $SaveDir `
    -Steps $Steps `
    -Resolution $Resolution `
    -Gamma $Gamma `
    -Temperature $Temperature

exit $LASTEXITCODE
