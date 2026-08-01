[CmdletBinding()]
param(
    [string]$Prompt = "Hello! Introduce yourself in one sentence.",
    [ValidateRange(1, 128)]
    [int]$MaxNewTokens = 12,
    [string]$Serial = "",
    [string]$AdbPath = "",
    [string]$DeviceRoot = "/data/local/tmp/mllm",
    [string]$OutputDir = "",
    [switch]$PushBinary
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

function Resolve-AdbExecutable {
    param([string]$ConfiguredPath)

    if ($ConfiguredPath) {
        if (-not (Test-Path -LiteralPath $ConfiguredPath -PathType Leaf)) {
            throw "ADB executable does not exist: $ConfiguredPath"
        }
        return (Resolve-Path -LiteralPath $ConfiguredPath).Path
    }

    $command = Get-Command adb.exe -ErrorAction SilentlyContinue
    if (-not $command) {
        $command = Get-Command adb -ErrorAction SilentlyContinue
    }
    if ($command) {
        return $command.Source
    }

    $knownPath = "C:\adb\adb.exe"
    if (Test-Path -LiteralPath $knownPath -PathType Leaf) {
        return $knownPath
    }

    throw "adb was not found. Add adb to PATH or pass -AdbPath C:\path\to\adb.exe"
}

function ConvertTo-ShellLiteral {
    param([string]$Value)

    $singleQuoteReplacement = "'" + '"' + "'" + '"' + "'"
    return "'" + $Value.Replace("'", $singleQuoteReplacement) + "'"
}

if ($DeviceRoot -notmatch '^/[A-Za-z0-9._/-]+$') {
    throw "DeviceRoot contains unsupported shell characters: $DeviceRoot"
}

$adb = Resolve-AdbExecutable $AdbPath

if (-not $OutputDir) {
    $OutputDir = Join-Path $PSScriptRoot "..\device-results"
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

if (-not $Serial) {
    $deviceLines = & $adb devices
    if ($LASTEXITCODE -ne 0) {
        throw "adb devices failed with exit code $LASTEXITCODE"
    }

    $onlineDevices = @(
        $deviceLines |
            ForEach-Object {
                if ($_ -match '^([^\s]+)\s+device$') { $Matches[1] }
            }
    )
    if ($onlineDevices.Count -eq 0) {
        throw "No authorized Android device is connected. Check USB debugging and adb devices."
    }
    if ($onlineDevices.Count -gt 1) {
        throw "Multiple Android devices are connected. Pass -Serial <device-serial>."
    }
    $Serial = $onlineDevices[0]
}

$adbPrefix = @("-s", $Serial)
$state = (& $adb @adbPrefix get-state 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $state -ne "device") {
    throw "Device $Serial is not ready: $state"
}

$deviceFiles = [ordered]@{
    Binary        = "$DeviceRoot/bin/demo_qwen_npu"
    Vocab         = "$DeviceRoot/vocab/qwen2.5_vocab.mllm"
    Merges        = "$DeviceRoot/vocab/qwen2.5_merges.txt"
    QnnModel      = "$DeviceRoot/models/Qwen2.5-1.5B-Instruct_rotated-noshadow.mllm"
    DecodingModel = "$DeviceRoot/models/Qwen2.5-1.5B-Instruct_rotated-Q40.mllm"
    QnnBackend    = "$DeviceRoot/qnn-lib/libQnnHtp.so"
    V79Skel       = "$DeviceRoot/qnn-lib/libQnnHtpV79Skel.so"
    CpuOpPackage  = "$DeviceRoot/qnn-lib/libQnnLLaMAPackage_CPU.so"
    HtpOpPackage  = "$DeviceRoot/qnn-lib/libQnnLLaMAPackage_HTP.so"
}

if ($PushBinary) {
    $localBinary = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\bin-arm-qnn\demo_qwen_npu"))
    if (-not (Test-Path -LiteralPath $localBinary -PathType Leaf)) {
        throw "Local demo binary does not exist: $localBinary"
    }
    Write-Host "[INFO] Pushing demo binary..."
    & $adb @adbPrefix push $localBinary $deviceFiles.Binary
    if ($LASTEXITCODE -ne 0) {
        throw "adb push failed with exit code $LASTEXITCODE"
    }
    & $adb @adbPrefix shell chmod 755 $deviceFiles.Binary
    if ($LASTEXITCODE -ne 0) {
        throw "chmod failed with exit code $LASTEXITCODE"
    }
}

foreach ($entry in $deviceFiles.GetEnumerator()) {
    & $adb @adbPrefix shell test -f $entry.Value
    if ($LASTEXITCODE -ne 0) {
        throw "Missing device file ($($entry.Key)): $($entry.Value)"
    }
}

$deviceBinDir = "$DeviceRoot/bin"
$deviceLog = "$deviceBinDir/simple_inference.log"
$deviceResult = "$deviceBinDir/simple_inference_result.txt"
$runTimestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$localLog = Join-Path $OutputDir "qwen_npu_${runTimestamp}_full.log"
$localResult = Join-Path $OutputDir "qwen_npu_${runTimestamp}_result.txt"
$quotedPrompt = ConvertTo-ShellLiteral $Prompt
$remoteCommand = @"
cd '$deviceBinDir' && LD_LIBRARY_PATH='$DeviceRoot/qnn-lib' ADSP_LIBRARY_PATH='$DeviceRoot/qnn-lib' ./demo_qwen_npu --vocab '$($deviceFiles.Vocab)' --merge '$($deviceFiles.Merges)' --qnn-model '$($deviceFiles.QnnModel)' --decoding-model '$($deviceFiles.DecodingModel)' --billion 1.5B-rotated --prompt $quotedPrompt --max-new-tokens $MaxNewTokens > '$deviceLog' 2>&1
"@.Trim()

Write-Host "[INFO] Device: $Serial"
Write-Host "[INFO] Prompt: $Prompt"
Write-Host "[INFO] Running Qwen2.5-1.5B on QNN HTP v79..."

& $adb @adbPrefix shell $remoteCommand
$inferenceExitCode = $LASTEXITCODE

$extractCommand = "sed -n '/^\[MLLM_RESULT_BEGIN\]$/,/^\[MLLM_RESULT_END\]$/p' '$deviceLog' > '$deviceResult'"
& $adb @adbPrefix shell $extractCommand
$extractExitCode = $LASTEXITCODE

& $adb @adbPrefix pull $deviceLog $localLog
if ($LASTEXITCODE -ne 0) {
    throw "Failed to pull the device log to: $localLog"
}

if ($extractExitCode -eq 0) {
    & $adb @adbPrefix pull $deviceResult $localResult
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to pull the concise Question/Answer result."
    }
}

if ($inferenceExitCode -ne 0) {
    Write-Host "[FAIL] Inference exited with code $inferenceExitCode" -ForegroundColor Red
    & $adb @adbPrefix shell tail -n 80 $deviceLog
    Write-Host "[INFO] Full log pulled to: $localLog"
    exit $inferenceExitCode
}

$proofCommand = "grep -a -E 'QNN Backend Build Id|QNN context retrieved|Number of HVX threads used|QNN accelerator \(execute\) time' '$deviceLog' | tail -n 12"
$proof = (& $adb @adbPrefix shell $proofCommand 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or $proof -notmatch 'Number of HVX threads used') {
    Write-Host "[FAIL] Process returned 0, but the log does not prove HTP execution." -ForegroundColor Red
    & $adb @adbPrefix shell tail -n 80 $deviceLog
    exit 3
}

Write-Host ""
Write-Host "QNN/NPU evidence:"
Write-Host $proof.TrimEnd()
Write-Host ""
Write-Host "Inference output (log tail):"
& $adb @adbPrefix shell tail -n 12 $deviceLog

if (Test-Path -LiteralPath $localResult -PathType Leaf) {
    $resultText = Get-Content -LiteralPath $localResult -Raw -Encoding UTF8
    if ($resultText.Trim()) {
        Write-Host ""
        Write-Host "Question/Answer:"
        Write-Host $resultText.Trim()
    } else {
        Write-Warning "The pulled Question/Answer result is empty. Use -PushBinary once to deploy the latest demo binary."
    }
}

Write-Host ""
Write-Host "[PASS] QNN HTP v79 execution completed (exit code 0)." -ForegroundColor Green
Write-Host "[INFO] Result pulled to: $localResult"
Write-Host "[INFO] Full log pulled to: $localLog"
Write-Host "[INFO] Device log: $deviceLog"
Write-Warning "Execution success does not prove answer quality; inspect the pulled result before acceptance."
exit 0
