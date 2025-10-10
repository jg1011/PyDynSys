#!/usr/bin/env pwsh
param(
  [string]$Pattern = "*.py",
  [switch]$StopOnFailure
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = "$($repoRoot)\src;$env:PYTHONPATH"

$files = Get-ChildItem -Path (Join-Path $repoRoot "examples") -Filter $Pattern -File | Sort-Object Name
if ($files.Count -eq 0) { Write-Error "No example .py files found"; exit 1 }

$passed = 0; $failed = 0; $exit = 0
foreach ($f in $files) {
  Write-Host "==> Running $($f.Name)"
  & python "$($f.FullName)"
  if ($LASTEXITCODE -eq 0) { $passed++ } else { $failed++; $exit = 1; if ($StopOnFailure) { break } }
  Write-Host
}
Write-Host "Summary: $passed passed, $failed failed"
exit $exit