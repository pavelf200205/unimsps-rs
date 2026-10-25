# Start manually in a second PowerShell window before a reconstruction.
# Uses vendor-neutral Windows performance counters. Does not change GPU settings.
param(
    [Parameter(Mandatory=$true)][string]$Out,
    [ValidateRange(1,86400)][int]$Seconds=600,
    [ValidateRange(1,60)][int]$Interval=2
)
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $Out) { throw "Output already exists: $Out" }
$paths=@(
    '\GPU Engine(*)\Utilization Percentage',
    '\GPU Adapter Memory(*)\Dedicated Usage',
    '\GPU Adapter Memory(*)\Shared Usage',
    '\GPU Process Memory(*)\Dedicated Usage',
    '\GPU Process Memory(*)\Shared Usage',
    '\Memory\Available Bytes'
)
# Resolve available sets individually: one unsupported/localized counter must
# not discard the other data. English counter names may need localization.
$available=@()
foreach ($path in $paths) {
    try { $null=Get-Counter -Counter $path -ErrorAction Stop; $available+=$path }
    catch { Write-Warning "Counter unavailable: $path" }
}
if (!$available.Count) { throw 'No requested counters are available on this Windows installation.' }
$first=$true
$stopAt=(Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $stopAt) {
    $sample=Get-Counter -Counter $available -SampleInterval $Interval -MaxSamples 1 -ErrorAction Continue
    if (!$sample) { continue }
    $names=@{}
    Get-Process -ErrorAction SilentlyContinue | ForEach-Object { $names[[int]$_.Id]=$_.ProcessName }
    $rows=foreach ($counter in $sample.CounterSamples) {
        $processId=$null
        if ($counter.InstanceName -match 'pid_(\d+)') { $processId=[int]$Matches[1] }
        [pscustomobject]@{
            Utc=$sample.Timestamp.ToUniversalTime().ToString('o')
            Counter=$counter.Path
            Instance=$counter.InstanceName
            ProcessId=$processId
            ProcessName=$(if ($null -ne $processId) {$names[$processId]} else {''})
            Value=$counter.CookedValue
            Status=$counter.Status
        }
    }
    if ($first) { $rows | Export-Csv -LiteralPath $Out -NoTypeInformation; $first=$false }
    else { $rows | Export-Csv -LiteralPath $Out -NoTypeInformation -Append }
}
