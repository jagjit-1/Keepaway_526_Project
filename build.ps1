# build.ps1 - Automated build script for Windows PowerShell
$msysBin = "C:\msys64\usr\bin"
$ucrtBin = "C:\msys64\ucrt64\bin"

if (Test-Path $msysBin) {
    $env:PATH = "$ucrtBin;$msysBin;" + $env:PATH
}

if ($args.Count -eq 0) {
    make
} else {
    make $args
}
