# Copying Content From Linux Command Line Into Markdown With Font Color

![image-20260117161900470](../../11-其他辅助工具/assets/image-20260117161900470.png)

Terminal applications that support rich copy can preserve ANSI colors when the output is pasted into an HTML-aware editor. Select the required lines, right-click, and choose **Copy as HTML**. Paste the result into an HTML block in Markdown when color is essential.

For portable documentation, prefer a fenced `text` or `console` block. It renders consistently in GitHub, static-site generators, and offline readers:

```console
$ python train.py
step=100 loss=0.182
```

Use colored HTML only for short diagnostic excerpts. Long logs should be stored as files and summarized in the chapter so that readers can find the relevant command, error, and resolution without scanning raw terminal output.
