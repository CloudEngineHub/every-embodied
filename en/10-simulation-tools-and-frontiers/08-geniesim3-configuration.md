# 1. Deploy the server side for openpi

GenieSim One-Click Start Video Tutorial: https://www.datawhale.cn/learn/content/258/6172


```bash
export OPENPI_ROOT="${OPENPI_ROOT:-$HOME/openpi}"
cd "$OPENPI_ROOT"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 uv run scripts/serve_policy.py --host='0.0.0.0' --port=8999 policy:checkpoint --policy.config=organize_items --policy.dir ./checkpoints/organize_items/29999
# If the environment is already prepared, run the server directly with Python:
python scripts/serve_policy.py --host='0.0.0.0' --port=8999 policy:checkpoint --policy.config=organize_items --policy.dir ./checkpoints/organize_items/29999
# `uv run` may create or refresh a project environment. Direct Python execution
# reuses the active openpi-server environment. OpenPI model files are cached in
# ~/.cache/openpi, so preserve that directory between sessions.
```


---

The appearance of this IP means success.

![image-20260203221823293](../../10-具身智能其他仿真工具及仿真前沿/assets/image-20260203221823293.png)

If there is a version error in av, change to pyav according to lerobot's dependency (this new version of the image has already fixed it for everyone).

There is one place that needs to be modified.

![image-20260203215608988](../../10-具身智能其他仿真工具及仿真前沿/assets/image-20260203215608988.png)

Need to modify pyproject.toml.

![image-20260203221805718](../../10-具身智能其他仿真工具及仿真前沿/assets/image-20260203221805718.png)

## 2. GenieSim Client

![image-20260115210912023](../../10-具身智能其他仿真工具及仿真前沿/assets/image-20260115210912023.png)

```
ssh -CNg -L 8999:127.0.0.1:8999 <user>@<policy-server-host>
# Enter the remote account password when prompted, or use an SSH key.
python -c "import socket; s=socket.socket(); s.connect(('localhost', 8999)); print('OK')"

```

![image-20260115210940126](../../10-具身智能其他仿真工具及仿真前沿/assets/image-20260115210940126.png)





Download assets (our mirror has been configured for you here; no additional configuration is needed):

```
clone下载到
genie_sim/source/geniesim/assets
https://modelscope.cn/datasets/agibot_world/GenieSimAssets

```

Set environment variables (already configured in the server image's .bashrc, no additional configuration is needed here)

```
# Project and runtime paths
export SIM_REPO_ROOT="${SIM_REPO_ROOT:-$HOME/genie_sim}"
export ISAAC_PYTHON="${ISAAC_PYTHON:-/isaac-sim/python.sh}"


# 5. 设置 API Key
export BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export API_KEY="sk-"
export MODEL="qwen3-max"
export VL_MODEL="qwen3-vl-plus"

# 6. 设置 Genie Sim 环境变量
export SIM_ASSETS=$SIM_REPO_ROOT/source/geniesim/assets

```

Then

```
cd "$SIM_REPO_ROOT"

"$ISAAC_PYTHON" source/geniesim/app/app.py --config source/geniesim/config/organize_items.yaml
# Or, when the image provides an omni_python wrapper:
omni_python source/geniesim/app/app.py --config source/geniesim/config/organize_items.yaml

```

You can see the actions being performed by the simulation renderer

## 3. Record the simulation results

Replace
`$SIM_REPO_ROOT/scripts/auto_record_and_extract.py`

For [auto_record_and_extract.py](../../10-具身智能其他仿真工具及仿真前沿/auto_record_and_extract.py) under the current tutorial folder

```
Edit `$SIM_REPO_ROOT/scripts/start_auto_record.sh`.

```

![image-20260204121620598](../../10-具身智能其他仿真工具及仿真前沿/assets/image-20260204121620598.png)

```
cd "$SIM_REPO_ROOT"

"$ISAAC_PYTHON" source/geniesim/app/app.py --config source/geniesim/config/organize_items_record.yaml
# Or:
omni_python source/geniesim/app/app.py --config source/geniesim/config/organize_items_record.yaml


```

Then open a new terminal.

Wait for the simulation frequency to appear

![image-20260204121858171](../../10-具身智能其他仿真工具及仿真前沿/assets/image-20260204121858171.png)

```
cd "$SIM_REPO_ROOT"
bash scripts/start_auto_record.sh
```

Then the cmap file will be automatically parsed and saved as an mp4 after the simulation ends.

```
ls "$SIM_REPO_ROOT/output/recording_data/recording_data/place_object_into_box_color/recording_20260204_121814/recording_20260204_121814_0.mcap" -lh
```





If mcap already exists

```
# Enter the project root.
cd "$SIM_REPO_ROOT"

# Point --bag_path to an existing recording directory.
python3 scripts/auto_record_and_extract.py \
  --bag_path "$SIM_REPO_ROOT/output/recording_data/recording_data/place_object_into_box_color/recording_20260204_141040"
```
