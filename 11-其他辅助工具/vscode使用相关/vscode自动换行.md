# 在 VS Code 中启用自动换行

自动换行只改变编辑器显示，不会在文件中插入换行符，适合阅读长日志、配置和 Markdown 段落。

## 图形界面设置

1. 按 `Ctrl+Shift+P` 打开命令面板。
2. 运行 **Preferences: Open User Settings**。
3. 搜索 `word wrap`，把 **Editor: Word Wrap** 设置为 `on`。

也可以直接打开用户配置文件并加入：

```json
{
  "editor.wordWrap": "on"
}
```

![VS Code 自动换行设置](./assets/image-20251216164240456.png)

## 验证与排错

打开一行较长的文本，缩窄编辑器窗口后应自动折行。若设置没有生效，检查工作区配置 `.vscode/settings.json` 是否把 `editor.wordWrap` 覆盖为 `off`；临时切换可使用 `Alt+Z`。
