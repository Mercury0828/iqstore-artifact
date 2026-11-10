# M1 submission wrapper (session loop or the Windows scheduled task "IQSTORE-M1-daily").
# Idempotent: submit_m1.py does at most one action per invocation (resume an outstanding job, or submit the
# epoch's job only if >= 110 min passed since the previous claim; v3 cadence) and enforces the approved tier (sha256 of qpu/M1_estimate.md, cap 600 s,
# whole-campaign projection). Polling it every 10 minutes is safe. The API token stays in ~/.qiskit/.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force (Join-Path $root 'runs') | Out-Null
$log = Join-Path $root 'runs\m1_daily.log'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONWARNINGS = 'ignore::DeprecationWarning'
$py = 'python'
"==== $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')) start" | Out-File -Append -Encoding utf8 $log
& $py collect/submit_m1.py --approved M1 --cap 600 *>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
"==== $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')) exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
