# Run manually. Alternates configurations to reduce ordering/temperature bias.
param(
    [Parameter(Mandatory=$true)][string]$Exe,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Data,
    [Parameter(Mandatory=$true)][string]$Out,
    [int]$Images = 6,
    [ValidateRange(1,100)][int]$Repeats = 3
)
$ErrorActionPreference = 'Stop'
$variants = @(
    @{ Name='reference'; Flags=@('--reference-encoder','--burn-softmax') },
    @{ Name='softmax_only'; Flags=@('--reference-encoder') },
    @{ Name='encoder_only'; Flags=@('--burn-softmax') },
    @{ Name='all'; Flags=@() }
)
$rows = @()
for ($repeat = 0; $repeat -lt $Repeats; $repeat++) {
    $order = if ($repeat % 2 -eq 0) { @(0,1,2,3) } else { @(3,2,1,0) }
    foreach ($index in $order) {
        $variant = $variants[$index]
        $destination = Join-Path $Out ("repeat_{0}_{1}" -f $repeat,$variant.Name)
        if (Test-Path -LiteralPath $destination) { throw "Output already exists: $destination" }
        $arguments = @($variant.Flags) + @('infer','--batch','--model',$Model,'--data',$Data,'--out',$destination,'--nb_img',"$Images")
        & $Exe @arguments
        if ($LASTEXITCODE -ne 0) { throw "Inference failed: $($variant.Name)" }
        foreach ($file in Get-ChildItem -LiteralPath $destination -Recurse -Filter metadata.json) {
            $metadata = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
            $rows += [pscustomobject]@{ Repeat=$repeat; Variant=$variant.Name; View=$file.Directory.Name; Seconds=$metadata.seconds }
        }
        $rows | Export-Csv -LiteralPath (Join-Path $Out 'timings.csv') -NoTypeInformation
    }
}
$rows | Group-Object Variant,View | ForEach-Object {
    $values = @($_.Group.Seconds | Sort-Object)
    $median = if ($values.Count % 2) { $values[[int][math]::Floor($values.Count/2)] } else { ($values[$values.Count/2-1]+$values[$values.Count/2])/2 }
    [pscustomobject]@{ ConfigurationView=$_.Name; MedianSeconds=$median; MinSeconds=$values[0]; MaxSeconds=$values[-1] }
} | Format-Table -AutoSize
