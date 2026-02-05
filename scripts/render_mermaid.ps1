param(
  [string]$InputPath = "docs/architecture.mmd",
  [string]$OutputPath = "docs/architecture.svg"
)

Write-Host "Rendering Mermaid diagram..."

if (-not (Test-Path $InputPath)) {
  Write-Host "Input file not found: $InputPath"
  exit 1
}

function Render-With-Mmdc {
  & mmdc -i $InputPath -o $OutputPath
  if ($LASTEXITCODE -ne 0) { return $false }
  return (Test-Path $OutputPath)
}

function Render-With-Npx {
  & npx --yes @mermaid-js/mermaid-cli -i $InputPath -o $OutputPath
  if ($LASTEXITCODE -ne 0) { return $false }
  return (Test-Path $OutputPath)
}

if (Get-Command mmdc -ErrorAction SilentlyContinue) {
  if (Render-With-Mmdc) {
    Write-Host "Wrote $OutputPath"
    exit 0
  } else {
    Write-Host "mmdc failed to render. Falling back to npx..."
  }
}

if (Get-Command npx -ErrorAction SilentlyContinue) {
  if (Render-With-Npx) {
    Write-Host "Wrote $OutputPath"
    exit 0
  } else {
    Write-Host "npx mermaid-cli failed. Check npm access or install mermaid-cli globally."
    exit 1
  }
}

Write-Host "Mermaid CLI not found."
Write-Host "Install with: npm install -g @mermaid-js/mermaid-cli"
exit 1
