# Front-end Display and Release Checklist

This chapter explains what the front-end of the household supplies assistant should display, as well as which files need to be checked before placing `tuntunclaw` in the public repository.

## Front-end Display Goals

The front-end does not need to be a complete product backend. Its goal is to allow users to see a clear demonstration of the closed loop:

1. Current material inventory
2. Low inventory status
3. Feishu reminder or table synchronization status
4. One-click ordering entry
5. Result of the latest task execution

## Front-end Display Block

- Task input area: Used for entering natural language tasks.
- Execution status area: Displays OpenClaw task stages.
- Inventory card area: Shows items, quantities, thresholds, and statuses.
- Lark synchronization area: Indicates whether synchronization is configured and if it was successful.
- Log area: Displays the latest execution and error messages.

## Pre-release Check

Before making the public `tuntunclaw` subdirectory, check the following:

- Do not submit `.env`
- Do not submit `sam_b.pt`
- Do not submit `temp/`, `trash/`, and `__pycache__/`
- Do not submit the `mask_*.png` configuration file
- Do not submit the scene editor: `scene_layout_editor.py` and `SCENE_LAYOUT_EDITOR_README.md`
- Do not submit the local Lark token, OpenAI key, and the robot's real internal network address

## Files that can be publicly retained

- `.env.example`
- `README.md`
- `requirements-py311.txt`
- `frontend/`
- `inventory.py`
- `integrations.py`
- `workflow_hooks.py`
- `main.py`
- `openclaw_like/`

## Verification Method

Run before submission:

```bash
git status --short
git ls-files | grep -E "sam_b.pt|\\.env$|scene_layout_editor|__pycache__|temp/|trash/"
```

The second command should not output any content.
