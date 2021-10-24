$location = Get-Location
Set-Location $PSScriptRoot
if (Get-Command 'npm' -ErrorAction SilentlyContinue) {
	Invoke-Expression "npm install"
	if (Get-Command "python" -ErrorAction SilentlyContinue) {
		Invoke-Expression "python -m venv .venv"
		Invoke-Expression ".venv/Scripts/activate"
		Invoke-Expression "pip install -r requirements.txt"
	}
	else {
		Write-Output "Python not installed"
	}
}
else {
	Write-Output "Node not installed"
}
Set-Location $location