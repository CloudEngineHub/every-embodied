# VR Remote Manipulation Python Library Usage Guide (Tested on Quest3)



The following example uses PowerShell and keeps the checkout under the current user's home directory. Set `VR_TELEOP_ROOT` before running it to use another location.

```powershell
if (-not $env:VR_TELEOP_ROOT) {
    $env:VR_TELEOP_ROOT = Join-Path $HOME "teleop_xr"
}

git clone https://github.com/qrafty-ai/teleop_xr.git $env:VR_TELEOP_ROOT
Set-Location $env:VR_TELEOP_ROOT

pip install -e .

# For IK support, install additional dependencies:
pip install "spatialmath-python>=1.1.15" "gitpython>=3.1.46" "xacro>=2.1.1" "filelock>=3.20.3" "viser>=1.0.21"
pip install git+https://github.com/chungmin99/pyroki.git
pip install git+https://github.com/chungmin99/ballpark.git

Set-Location webxr
npm install
npm run build

Set-Location $env:VR_TELEOP_ROOT
python -m teleop_xr.demo # Tests the data transport path only.

mamba activate py310
Set-Location $env:VR_TELEOP_ROOT

python -m teleop_xr.demo --mode ik --robot-class so101 --input-mode CONTROLLER --host 0.0.0.0 --port 4444 --no-tui

python -m teleop_xr.demo --mode ik --robot-class franka --input-mode CONTROLLER --host 0.0.0.0 --port 4444 --no-tui


Put on the VR headset after the service starts. Hold both Grip buttons beside the thumbs to enable hand-motion control; releasing either button pauses control. Double-click twice to reset.

# Stop the process bound to port 4444.
Get-NetTCPConnection -LocalPort 4444 -ErrorAction SilentlyContinue | Select-Object -Expand OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force }
```
