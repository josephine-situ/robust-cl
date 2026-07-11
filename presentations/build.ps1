# Compile every *.tex in presentations/ into presentations/build/
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$build = Join-Path $here "build"
New-Item -ItemType Directory -Force -Path $build | Out-Null
Push-Location $here
try {
    foreach ($tex in Get-ChildItem -Path $here -Filter *.tex) {
        pdflatex -interaction=nonstopmode -output-directory=build $tex.Name | Out-Null
        pdflatex -interaction=nonstopmode -output-directory=build $tex.Name | Out-Null
        Write-Host "Wrote $(Join-Path $build ($tex.BaseName + '.pdf'))"
    }
} finally {
    Pop-Location
}
