# Compile the manuscript in an ASCII path, because the workspace path contains
# characters that pdflatex cannot pass through TEXMF_OUTPUT_DIRECTORY.
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [string]$Build = "C:\vcam_build"
)

$env:PATH = "C:\Users\24481084\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\TinyTeX\bin\windows;$env:PATH"

Remove-Item -Recurse -Force $Build -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $Build | Out-Null
Copy-Item "$Source\*" $Build -Recurse -Force
Set-Location $Build

foreach ($doc in @("main", "supplement")) {
    Write-Output "=== $doc ==="
    latexmk -pdf -interaction=nonstopmode -f "$doc.tex" *> "$doc.build.log"
    $log = Get-Content "$doc.log" -Raw -ErrorAction SilentlyContinue
    if ($null -eq $log) { Write-Output "NO LOG"; continue }
    $errors = Select-String -Path "$doc.log" -Pattern '^!|LaTeX Error|Undefined control sequence|Citation .* undefined|Reference .* undefined|Overfull \\hbox \(([3-9][0-9]|[0-9]{3,})' -AllMatches
    if ($errors) {
        Write-Output "--- issues ---"
        $errors | ForEach-Object { $_.Line } | Select-Object -First 40
    } else {
        Write-Output "no errors, no undefined references, no overfull boxes above 30pt"
    }
    if (Test-Path "$doc.pdf") {
        $pages = (Select-String -Path "$doc.log" -Pattern 'Output written on .*\((\d+) pages').Matches.Groups[1].Value
        Write-Output "PDF built: $pages pages"
    } else {
        Write-Output "PDF MISSING"
    }
}
