# 在现有 WSA 环境中安装 APK

本页适用于已经能够启动 Windows Subsystem for Android 的电脑。可以使用图形化安装器，也可以通过 Android 调试桥安装 APK。

## 图形化安装器

- https://apps.microsoft.com/detail/9nn9k1426pbm?referrer=psi&hl=zh-CN&gl=HK
- https://apps.microsoft.com/detail/9nvh3mgqncn7?hl=en-US&gl=US

安装后选择本地 APK 文件，并按界面提示授权 WSA 调试连接。

## 命令行安装

先在 WSA 设置中开启开发者模式并查看调试地址，然后执行：

```powershell
adb connect 127.0.0.1:58526
adb install .\example.apk
```

## 验证与排错

```powershell
adb devices
adb shell pm list packages
```

设备列表应显示已连接实例，包列表中应出现新应用。若连接被拒绝，先启动 WSA 并重新开启开发者模式；若安装提示签名冲突，卸载旧版本后再安装同一来源的新版本。
