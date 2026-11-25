param(
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [Parameter(Mandatory = $true)]
    [long]$ExpectedBytes,

    [ValidateRange(1, 64)]
    [int]$ChunkSizeMB = 8
)

$ErrorActionPreference = "Stop"

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

if (Test-Path -LiteralPath $OutputPath) {
    $currentBytes = (Get-Item -LiteralPath $OutputPath).Length
    if ($currentBytes -gt $ExpectedBytes) {
        throw "Existing file is larger than expected: $currentBytes > $ExpectedBytes"
    }
} else {
    $currentBytes = 0
}

$chunkBytes = [long]$ChunkSizeMB * 1MB
$temporaryPath = "$OutputPath.chunk"
$lastReportedBucket = -1

try {
    while ($currentBytes -lt $ExpectedBytes) {
        $rangeEnd = [Math]::Min($currentBytes + $chunkBytes - 1, $ExpectedBytes - 1)
        $requiredBytes = $rangeEnd - $currentBytes + 1

        if (Test-Path -LiteralPath $temporaryPath) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }

        & curl.exe -L --fail --silent --show-error `
            --connect-timeout 20 --max-time 60 `
            --range "$currentBytes-$rangeEnd" `
            -o $temporaryPath $Url

        $curlExitCode = $LASTEXITCODE
        $receivedBytes = if (Test-Path -LiteralPath $temporaryPath) {
            (Get-Item -LiteralPath $temporaryPath).Length
        } else {
            0
        }

        if ($curlExitCode -ne 0 -or $receivedBytes -ne $requiredBytes) {
            Write-Warning "Chunk $currentBytes-$rangeEnd failed (curl=$curlExitCode, bytes=$receivedBytes/$requiredBytes); retrying."
            if (Test-Path -LiteralPath $temporaryPath) {
                Remove-Item -LiteralPath $temporaryPath -Force
            }
            Start-Sleep -Seconds 1
            continue
        }

        $destination = [System.IO.File]::Open(
            $OutputPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::Read
        )
        $source = $null
        try {
            [void]$destination.Seek($currentBytes, [System.IO.SeekOrigin]::Begin)
            $source = [System.IO.File]::OpenRead($temporaryPath)
            $source.CopyTo($destination)
            $destination.Flush()
        } finally {
            if ($null -ne $source) {
                $source.Dispose()
            }
            $destination.Dispose()
        }

        Remove-Item -LiteralPath $temporaryPath -Force
        $currentBytes += $requiredBytes

        $progressBucket = [Math]::Floor(($currentBytes * 20.0) / $ExpectedBytes)
        if ($progressBucket -ne $lastReportedBucket -or $currentBytes -eq $ExpectedBytes) {
            $percent = [Math]::Round(($currentBytes * 100.0) / $ExpectedBytes, 1)
            Write-Output "$percent% ($currentBytes/$ExpectedBytes bytes)"
            $lastReportedBucket = $progressBucket
        }
    }
} finally {
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Force
    }
}

Write-Output "Completed: $OutputPath"
