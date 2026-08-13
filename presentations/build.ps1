# Compile every *.tex in presentations/.
# Output locations come from presentations/.latexmkrc: the .pdf (and .synctex.gz)
# land here next to the .tex, every other artifact goes to presentations/build/.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $here
try {
    foreach ($tex in Get-ChildItem -Path $here -Filter *.tex) {
        latexmk -pdf -synctex=1 -interaction=nonstopmode $tex.Name | Out-Null
        Write-Host "Wrote $(Join-Path $here ($tex.BaseName + '.pdf'))"
    }
} finally {
    Pop-Location
}
