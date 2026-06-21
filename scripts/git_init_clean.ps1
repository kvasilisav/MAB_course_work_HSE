$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .git)) {
    git init
    Write-Host "Created .git"
}

$legacy = @(
    "docs/literature_review.md",
    "docs/repo_plan_llm.md",
    "docs/coursework_roadmap.md",
    "docs/experiments_worklog.md",
    "docs/product_decision_matrix.md",
    "docs/papers.csv",
    "docs/datasets_catalog.md",
    "docs/glossary.md",
    "docs/coursework_full.md",
    "docs/coursework_full.docx",
    "docs/scenarios/product_ab_pair.md",
    "docs/internal",
    "scripts/assemble_coursework_md.py"
)
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
foreach ($p in $legacy) {
    git ls-files --error-unmatch $p 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        git rm -r --cached $p | Out-Null
    }
}
$ErrorActionPreference = $prevEap

git add .gitignore README.md run_all.ps1 run_all.sh pyproject.toml requirements.txt
git add src tests scripts notebooks
git add docs/front_matter.md docs/literature_review_academic.md
git add docs/experiments_chapter.md docs/conclusion_chapter.md docs/appendix.md
git add data/raw/.gitkeep data/processed/.gitkeep
git add outputs/figures/*.png
git add outputs/extended_full/*_summary.csv outputs/extended_full/full_report.json 2>$null
git add outputs/obd_pair/*_summary.csv outputs/obd_pair/pair_selection.csv 2>$null
git add outputs/ab_validity/summary_h20000.csv 2>$null
git add outputs/inference_valid/e11_summary.csv 2>$null
git add outputs/sequential_valid/e14_summary.csv 2>$null

Write-Host ""
Write-Host "Staged. Review: git status"
Write-Host "Commit when ready: git commit -m ""MAB vs A/B coursework: code + academic sources"""
Write-Host ""
Write-Host "Not in git: docs/internal/, outputs/* except figures/*.png and summary CSVs, OBD CSV"
