$ErrorActionPreference = 'Stop'

$destination = 'D:\IC Mres AIML\Probabilistic memory states for modern RNNs\kronecker+s2vgp\ICLR Formal experiment\ERA5_experiment_summary_20260826'
$dFormal = 'D:\IC Mres AIML\Probabilistic memory states for modern RNNs\kronecker+s2vgp\ICLR Formal experiment'
$wslRepo = '\\wsl.localhost\Ubuntu-24.04\home\zd929\projects\stvgp_kronecker'
$sourceReport = Join-Path $wslRepo 'results\experiments_era5_ohsvgp_heldout_fullspace\paper_ready\ICLR Formal experiment'
$sourceDiagnostics = Join-Path $wslRepo 'results\diagnostics'

New-Item -ItemType Directory -Force -Path $destination | Out-Null

function Copy-Tree([string]$source, [string]$target) {
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Missing source: $source"
    }
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Copy-Item -Path (Join-Path $source '*') -Destination $target -Recurse -Force
}

function Copy-FileSet([string]$sourceDir, [string]$targetDir, [string]$pattern) {
    if (-not (Test-Path -LiteralPath $sourceDir)) {
        throw "Missing source directory: $sourceDir"
    }
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    Get-ChildItem -LiteralPath $sourceDir -File -Filter $pattern | Copy-Item -Destination $targetDir -Force
}

# Core formal archives already under ICLR Formal experiment.
Copy-Tree (Join-Path $dFormal 'iclr_era5_stage2plus') (Join-Path $destination '01_primary_stage2plus')
Copy-Tree (Join-Path $dFormal 'iclr_era5_full_benchmark') (Join-Path $destination '02_valid_direct_target_full_benchmark')

# The paper-ready tree is curated evidence. Exclude its duplicate full benchmark
# and the explicitly smoke-only baseline; both decisions are recorded below.
$paperReadyTarget = Join-Path $destination '03_paper_ready_era5_package'
New-Item -ItemType Directory -Force -Path $paperReadyTarget | Out-Null
Get-ChildItem -LiteralPath $sourceReport -Directory |
    Where-Object { $_.Name -notin @('iclr_era5_full_benchmark', 'stvgp_era5_baseline_smoke') } |
    ForEach-Object { Copy-Tree $_.FullName (Join-Path $paperReadyTarget $_.Name) }

# Valid diagnostic families cited by the audit document.
$diagnosticTarget = Join-Path $destination '04_valid_diagnostics'
Copy-Tree (Join-Path $sourceDiagnostics 'routeb_task1_10_variance') (Join-Path $diagnosticTarget 'routeb_task1_10_variance')
Copy-Tree (Join-Path $sourceDiagnostics 'routeb_task1_10_vfe_budget_comparison') (Join-Path $diagnosticTarget 'routeb_task1_10_vfe_budget_comparison')
Copy-Tree (Join-Path $sourceDiagnostics 'routeb_efficiency_optimization') (Join-Path $diagnosticTarget 'routeb_efficiency_optimization')

# Reproduction code and cloud launch configuration, without copying the raw ERA5
# dataset. The source dataset location and protocol are recorded in the manifest.
$reproTarget = Join-Path $destination '05_reproduction_and_configs'
Copy-FileSet (Join-Path $wslRepo 'scripts') (Join-Path $reproTarget 'scripts') '*era5*'
Copy-Tree (Join-Path $wslRepo 'cloud\autodl_era5') (Join-Path $reproTarget 'cloud_autodl_era5')
Copy-Tree (Join-Path $wslRepo 'configs') (Join-Path $reproTarget 'configs')

# Audit sources are copied as provenance documents.
$auditTarget = Join-Path $destination '00_audit_sources'
New-Item -ItemType Directory -Force -Path $auditTarget | Out-Null
Copy-Item -LiteralPath (Join-Path $dFormal 'ERA5_EXPERIMENT_AUDIT_AND_THESIS_MAP.md') -Destination $auditTarget -Force
Copy-Item -LiteralPath (Join-Path $dFormal 'EXPERIMENT_RESULT_MASTER_AUDIT_AND_THESIS_MAP.md') -Destination $auditTarget -Force

# Capture the current WSL repository identity without changing its dirty worktree.
$commit = (& wsl.exe -d Ubuntu-24.04 -- bash -lc "cd /home/zd929/projects/stvgp_kronecker && git rev-parse HEAD").Trim()
$statusLines = @(& wsl.exe -d Ubuntu-24.04 -- bash -lc "cd /home/zd929/projects/stvgp_kronecker && git status --short")

$decisions = @(
    [PSCustomObject]@{category='primary'; name='iclr_era5_stage2plus'; status='incomplete_but_retained'; location='01_primary_stage2plus'; reason='Preferred five-seed modern archive; audit reports 129/130 complete records and Maddox long-stream seed 0 missing.'},
    [PSCustomObject]@{category='formal'; name='iclr_era5_full_benchmark'; status='valid_separate_protocol'; location='02_valid_direct_target_full_benchmark'; reason='Complete declared three-seed direct-target/resource benchmark; not numerically poolable with stage2plus.'},
    [PSCustomObject]@{category='paper_ready'; name='paper_ready_era5_package'; status='valid_or_audited'; location='03_paper_ready_era5_package'; reason='Curated paper-ready evidence; duplicate full benchmark omitted because it is retained in section 02.'},
    [PSCustomObject]@{category='diagnostic'; name='routeb_task1_10_variance'; status='valid_diagnostic'; location='04_valid_diagnostics/routeb_task1_10_variance'; reason='Predictive-variance and coverage-by-task evidence cited by the audit.'},
    [PSCustomObject]@{category='diagnostic'; name='routeb_task1_10_vfe_budget_comparison'; status='valid_diagnostic'; location='04_valid_diagnostics/routeb_task1_10_vfe_budget_comparison'; reason='Finite-DTC versus VFE budget/convergence evidence cited by the audit.'},
    [PSCustomObject]@{category='diagnostic'; name='routeb_efficiency_optimization'; status='valid_appendix_diagnostic'; location='04_valid_diagnostics/routeb_efficiency_optimization'; reason='Solver/operator and runtime evidence; implementation-specific appendix material.'},
    [PSCustomObject]@{category='excluded'; name='stvgp_era5_baseline_smoke'; status='excluded'; location='not copied'; reason='Smoke-only performance is not evidence under the audit rules.'},
    [PSCustomObject]@{category='excluded'; name='paper_ready/iclr_era5_full_benchmark'; status='deduplicated'; location='02_valid_direct_target_full_benchmark'; reason='Same package retained once in section 02.'},
    [PSCustomObject]@{category='excluded'; name='results/*smoke/*probe/*tmp*'; status='excluded'; location='not copied'; reason='Implementation checks or exploratory runs; no performance ranking.'}
)

$metadata = [ordered]@{
    archive_name = 'ERA5 experiment summary'
    archive_date = '2026-08-26'
    current_repo_commit = $commit
    current_repo_worktree_dirty = ($statusLines.Count -gt 0)
    current_repo_status_lines = $statusLines
    destination = $destination
    source_roots = [ordered]@{
        formal_iclr = $dFormal
        wsl_repository = $wslRepo
        paper_ready = $sourceReport
        diagnostics = $sourceDiagnostics
        dataset = (Join-Path $wslRepo 'stvgp_kronecker\data\era5\processed_timeseries_4')
    }
    protocol_summary = [ordered]@{
        primary_archive = 'iclr_era5_stage2plus'
        data = 'ERA5-Land processed_timeseries_4, variable index 0, 1000 fixed UK locations'
        split = '800 train, 80 validation within train, 720 fit, 200 held-out test; seeds 0-4'
        calibration = 'Task 1, 186 hourly observations'
        streams = 'Task 2: 186 hours/19 blocks; Tasks 2-10: 1674 hours/171 task-aware blocks'
        xlag = '133 features, lag length 10, ridge 0.001'
        hippo = 'analytic HiPPO representation with 256 RFF samples'
        primary_provenance_caveat = 'stage2plus metadata identifies finite DTC; current thesis presents VFE, so do not describe it as a matched VFE result without a rerun'
    }
    evidence_rule = 'Per-seed archives/manifests take priority over aggregate reports; incomplete and failed records are retained as status evidence, not promoted to the main table.'
    decisions = $decisions
}
$metadata | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $destination 'archive_manifest.json') -Encoding UTF8
$decisions | Export-Csv -LiteralPath (Join-Path $destination 'experiment_index.csv') -NoTypeInformation -Encoding UTF8

$readme = @"
# ERA5 experiment summary

Snapshot date: 2026-08-26
Repository commit: $commit

This archive consolidates the ERA5 evidence referenced by the two audit documents. Original source directories are not modified. Incomplete or failed records are retained with an explicit status; they are not silently promoted to the paper's main table.

## Contents

- `01_primary_stage2plus/`: preferred modern five-seed archive. The audit finds 129/130 complete records; Maddox StreamingSGPR Tasks 2--10 seed 0 is missing.
- `02_valid_direct_target_full_benchmark/`: separate complete three-seed direct-target/resource benchmark. Do not pool its numbers with stage2plus.
- `03_paper_ready_era5_package/`: curated paper-ready historical evidence, including parity, model-design, baseline, and taskwise diagnostics. The nested duplicate full benchmark is omitted.
- `04_valid_diagnostics/`: variance, finite-DTC/VFE budget, and efficiency/operator diagnostics cited by the audit.
- `05_reproduction_and_configs/`: ERA5 scripts, AutoDL launch configuration, and repository configuration files. Raw ERA5 data remains at the recorded source path.
- `00_audit_sources/`: the two source audit documents.
- `experiment_index.csv`: inclusion, status, and handling decision for each family.
- `archive_manifest.json`: protocol, provenance, and current repository identity.
- `file_manifest.csv`: SHA-256 manifest for every archived file.

## Interpretation boundary

The primary stage2plus archive is the preferred evidence for the modern long strict-online comparison, but its solver metadata says finite DTC while the current thesis presents VFE. Preserve that caveat next to any thesis table. The direct-target benchmark, paper-ready historical package, and diagnostics use different protocols or purposes and must be reported separately.
"@
Set-Content -LiteralPath (Join-Path $destination 'README.md') -Value $readme -Encoding UTF8

$allFiles = Get-ChildItem -LiteralPath $destination -File -Recurse | Where-Object { $_.Name -notin @('file_manifest.csv') }
$allFiles | ForEach-Object {
    [PSCustomObject]@{
        relative_path = $_.FullName.Substring($destination.Length).TrimStart('\')
        bytes = $_.Length
        last_write_time = $_.LastWriteTime.ToString('o')
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
} | Export-Csv -LiteralPath (Join-Path $destination 'file_manifest.csv') -NoTypeInformation -Encoding UTF8

Write-Output "ARCHIVE=$destination"
Write-Output "COMMIT=$commit"
Write-Output "FILES=$((Get-ChildItem -LiteralPath $destination -File -Recurse).Count)"
Write-Output "BYTES=$(((Get-ChildItem -LiteralPath $destination -File -Recurse | Measure-Object -Property Length -Sum).Sum))"
