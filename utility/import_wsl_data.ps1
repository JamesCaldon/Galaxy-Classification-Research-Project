$baseNames = (Get-ChildItem -Path "\\wsl$\Ubuntu\home\james\generators\").BaseName
#TODO: Copy Text file with parameters information

$disc_size = "fd=0.3-0.9_discs_orig_200"
foreach ($baseName in $baseNames) {
    $dest = "$PSScriptRoot\..\data\raw\training\"
    if ($baseName.Contains("disc_1"))
    {
        $dest = "$dest\$disc_size\disc"
    } else 
    {
        $dest = "$dest\$disc_size\no_disc"
    }
    
    $directory = New-Item -Name $baseName $dest -ItemType directory -Force
    
    $item = Copy-Item "\\wsl$\Ubuntu\home\james\generators\$baseName\work.dir\*" $directory -PassThru
    Write-Host $item
    Write-Host $directory
    tar -xf $item -C $directory
    Move-Item "$directory\m1.dir\*" $directory -Force
    Remove-Item "$directory\m1.dir" -Force
}