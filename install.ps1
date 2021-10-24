if (Get-Command 'npm' --errorAction SilentlyContinue) {
	Invoke-Expression "npm install"
	if (Get-Command "python" --errorAction SilentlyContinue) {
		Invoke-Expression "python -m venv .venv"
		Invoke-Expression "pip install -r requirements.txt"
	}
	else {
		Write-Output "Python not installed"
	}
}
else {
	Write-Output "Node not installed"
}