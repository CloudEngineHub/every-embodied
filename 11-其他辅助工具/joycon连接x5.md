# 在 RDK X5 上安装 Joy-Con 内核驱动

Joy-Con 驱动需要针对开发板当前运行的 Linux 内核编译。开始前先准备驱动源码、编译工具和与运行内核完全一致的头文件。

## 检查内核与头文件

```bash
KERNEL_VERSION="$(uname -r)"
echo "$KERNEL_VERSION"
ls "/usr/src/linux-headers-$KERNEL_VERSION"
```

如果头文件目录存在，但 `/lib/modules` 中缺少构建链接，可以创建链接：

```bash
sudo mkdir -p "/lib/modules/$KERNEL_VERSION"
sudo ln -fs "/usr/src/linux-headers-$KERNEL_VERSION" "/lib/modules/$KERNEL_VERSION/build"
sudo ln -fs "/usr/src/linux-headers-$KERNEL_VERSION" "/lib/modules/$KERNEL_VERSION/source"
```

## 编译并安装

在 `joycon-robotics` 源码目录执行：

```bash
cd /path/to/joycon-robotics
make clean
make
sudo make install --kernelsourcedir "/usr/src/linux-headers-$KERNEL_VERSION"
sudo depmod -a
```

## 验证与排错

连接手柄后运行：

```bash
lsmod | grep -E 'hid_nintendo|joycon'
sudo dmesg --follow
```

日志中应出现新输入设备。若编译提示版本不匹配，重新确认 `uname -r` 与头文件目录名称一致；若设备已配对但没有输入事件，检查驱动模块是否加载以及当前用户是否有读取 `/dev/input` 的权限。
