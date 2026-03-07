param(
    [string]$OutDir = (Join-Path $PSScriptRoot "dist")
)

$ErrorActionPreference = "Stop"

function Resolve-TexEngine {
    param(
        [string]$CommandName
    )

    $candidate = Get-Command $CommandName -ErrorAction SilentlyContinue
    if ($candidate) {
        return $candidate.Source
    }

    $texLivePath = Join-Path "C:\texlive\2026\bin\windows" ($CommandName + ".exe")
    if (Test-Path $texLivePath) {
        return $texLivePath
    }

    throw "TeX engine '$CommandName' was not found."
}

function Invoke-TexBuild {
    param(
        [string]$EnginePath,
        [string]$WorkingDir
    )

    Push-Location $WorkingDir
    try {
        & $EnginePath -interaction=nonstopmode -halt-on-error main.tex
        & $EnginePath -interaction=nonstopmode -halt-on-error main.tex
    }
    finally {
        Pop-Location
    }
}

$pdfLatex = Resolve-TexEngine -CommandName "pdflatex"
$luaLatex = Resolve-TexEngine -CommandName "lualatex"

$enDir = Join-Path $PSScriptRoot "en"
$jaDir = Join-Path $PSScriptRoot "ja"

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

Invoke-TexBuild -EnginePath $pdfLatex -WorkingDir $enDir
Invoke-TexBuild -EnginePath $luaLatex -WorkingDir $jaDir

$enOut = Join-Path $OutDir "audio-chain-physics-paper-en.pdf"
$jaOut = Join-Path $OutDir "audio-chain-physics-paper-ja.pdf"

Copy-Item -Path (Join-Path $enDir "main.pdf") -Destination $enOut -Force
Copy-Item -Path (Join-Path $jaDir "main.pdf") -Destination $jaOut -Force

Write-Output "Built:"
Write-Output "  $enOut"
Write-Output "  $jaOut"
