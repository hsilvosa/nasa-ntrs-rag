param(
    [ValidateSet("qwen3-4b", "gemma4-12b", "qwen3-14b")]
    [string]$Model = "qwen3-4b",
    [int]$Port = 8080,
    [int]$ContextSize = 8192,
    [int]$GpuLayers = 99
)

$ModelMap = @{
    "qwen3-4b"   = "D:\models\own\qwen3-4b-instruct-2507-base\qwen3-4b-instruct-2507-base-Q4_K_M.gguf"
    "gemma4-12b" = "D:\models\core\gemma4-12b-qat-base\gemma-4-12b-it-qat-q4_0.gguf"
    "qwen3-14b"  = (Get-ChildItem "D:\models\core\qwen3-14b-base\*.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName)
}

$ModelPath = $ModelMap[$Model]
if (-not $ModelPath -or -not (Test-Path $ModelPath)) {
    Write-Error "Model file not found for '$Model': $ModelPath"
    exit 1
}

Write-Host "Starting llama-server on http://127.0.0.1:$Port/v1" -ForegroundColor Cyan
Write-Host "Model: $ModelPath (ctx=$ContextSize, ngl=$GpuLayers)" -ForegroundColor Green

& llama-server `
    --model $ModelPath `
    --alias $Model `
    --host 127.0.0.1 `
    --port $Port `
    --ctx-size $ContextSize `
    --n-gpu-layers $GpuLayers `
    --flash-attn on
