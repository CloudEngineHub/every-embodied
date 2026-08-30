# Vscode Auto Line Wrapping

Use `Ctrl+Shift+P` to open the Command Palette, select **Preferences: Open User Settings**, and search for `word wrap`. Set **Editor: Word Wrap** to `on`.

The equivalent JSON setting is:

```json
{
  "editor.wordWrap": "on",
  "editor.wordWrapColumn": 100
}
```

`on` wraps at the viewport edge. Use `wordWrapColumn` when documentation should wrap at a stable column while editing. This changes only the editor display; it does not insert line breaks or modify the Markdown file.

![image-20251216164240456](../../../11-其他辅助工具/vscode使用相关/assets/image-20251216164240456.png)
