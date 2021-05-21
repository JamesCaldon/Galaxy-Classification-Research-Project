$baseNames = (Get-ChildItem -Path "\\wsl$\Ubuntu\home\james\generators\").BaseName
#TODO: Copy Text file with parameters information

$disc_size = "small_medium_discs"
foreach ($baseName in $baseNames) {
    $dest = "C:\Users\James_Dev_Account\OneDrive - The University of Western Australia\Documents\Honours - Galaxy Classification\Galaxy-Classification-Research-Project\generated_data"
    if ($baseName.Contains("disc_1"))
    {
        $dest = "$dest\disc\$disc_size"
    } else 
    {
        $dest = "$dest\no_disc\$disc_size"
    }
    $directory = New-Item -Name $baseName $dest -ItemType directory -Force
    
    $item = Copy-Item "\\wsl$\Ubuntu\home\james\generators\$baseName\work.dir\*" $directory -PassThru
    Write-Host $item
    Write-Host $directory
    tar -xf $item -C $directory
    Move-Item "$directory\m1.dir\*" $directory -Force
    Remove-Item "$directory\m1.dir" -Force
}