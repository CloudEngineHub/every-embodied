# Ubuntu 安装 IBus 中文拼音输入法

本页使用 IBus 输入法框架安装智能拼音，适用于 Ubuntu 桌面环境。

## 安装与配置

```bash
sudo apt update
sudo apt install -y ibus ibus-libpinyin
ibus restart

ibus-setup
```

在弹出的窗口中操作



输入上述命令后，会弹出一个名为 "IBus Preferences" 的窗口：

1. 点击顶部的 **Input Method (输入法)** 标签页。
2. 点击右侧的 **Add (添加)** 按钮。
3. 在列表中依次选择：**Chinese (中文)** -> **Intelligent Pinyin (智能拼音)** -> **Add (添加)**。

注销并重新登录桌面会话，然后使用系统输入法快捷键切换到智能拼音。

## 验证与排错

```bash
ibus list-engine | grep -i pinyin
```

命令应列出拼音引擎。若设置窗口中没有中文选项，重新安装 `ibus-libpinyin` 并注销登录；若能选择但无法输入，运行 `ibus-daemon -drx` 后再次测试，并确认桌面会话使用 IBus 作为输入法框架。
