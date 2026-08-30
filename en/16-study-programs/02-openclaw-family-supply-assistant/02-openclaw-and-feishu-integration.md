# OpenClaw and Feishu Integration

This chapter only covers the connection link, without discussing the UI or the subsequent inventory logic.

## After completing this chapter

Readers should know:

1. What configurations does OpenClaw read?
2. Where are the SHEET app credentials stored?
3. Why do we need to write a separate layer of configuration for multi-dimensional table synchronization?

## Reading Order

The integration layer of OpenClaw first reads the environment variables, and then falls back to the local `~/.openclaw/openclaw.json`.

Priority is as follows:

1. `OPENCLAW_FEISHU_APP_ID`
2. `OPENCLAW_FEISHU_APP_SECRET`
3. `OPENCLAW_FEISHU_NOTIFY_TARGET`
4. `OPENCLAW_FEISHU_NOTIFY_RECEIVE_ID_TYPE`
5. `OPENCLAW_FEISHU_BITABLE_APP_TOKEN`
6. `OPENCLAW_FEISHU_BITABLE_TABLE_ID`

## What needs to be prepared

- Application credentials in the Feishu Open Platform
- A writable OpenClaw local configuration file
- Enabled multi-dimensional table permissions

## Key Checkpoints

- If `appId` and `appSecret` are empty, both notification and synchronization will fail.
- If only the notification target is configured, but `appToken/tableId` with a multi-dimensional table is not configured, inventory synchronization will not start.
- If the configuration file contains a Windows-written UTF-8 BOM, reading JSON with Python may fail. It is recommended to write it as UTF-8 without BOM.

## Corresponding Code

- [integrations.py](../../../16-专题组队学习/02-OpenClaw家庭物资助手/tuntunclaw/integrations.py)
- [workflow_hooks.py](../../../16-专题组队学习/02-OpenClaw家庭物资助手/tuntunclaw/workflow_hooks.py)
- [main.py](../../../16-专题组队学习/02-OpenClaw家庭物资助手/tuntunclaw/main.py)
