Write-Host "Cleaning Python/test artifacts..." -ForegroundColor Cyan

$paths = @(".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov", "dist", "build")
foreach ($p in $paths) {
  if (Test-Path $p) { Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue }
}

if (Test-Path ".coverage") { Remove-Item ".coverage" -Force -ErrorAction SilentlyContinue }
if (Test-Path "coverage.xml") { Remove-Item "coverage.xml" -Force -ErrorAction SilentlyContinue }

Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | ForEach-Object {
  Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
}

Get-ChildItem -Recurse -File -Filter "*.pyc" -ErrorAction SilentlyContinue | ForEach-Object {
  Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
}

Write-Host "Done." -ForegroundColor Green
