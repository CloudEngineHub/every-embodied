# Joycon Connects to X5

This note covers the kernel-module step required when a Joy-Con driver must be built against the RDK X5 kernel. Confirm the running kernel and available headers before installation:

```bash
uname -r
ls /usr/src
```

Build the driver with the header directory that matches `uname -r`:

```bash
cd joycon-robotics/
export KERNEL_VERSION="$(uname -r)"
export KERNEL_HEADERS="/usr/src/linux-headers-$KERNEL_VERSION"
test -d "$KERNEL_HEADERS"

sudo mkdir -p "/lib/modules/$KERNEL_VERSION"
sudo ln -fs "$KERNEL_HEADERS" "/lib/modules/$KERNEL_VERSION/build"
sudo ln -fs "$KERNEL_HEADERS" "/lib/modules/$KERNEL_VERSION/source"
make install --kernelsourcedir "$KERNEL_HEADERS"
```

Reconnect the controller after installation, then inspect `dmesg` and the input-device list. A build for a different kernel version can install successfully but will not load; rebuild the module whenever the board kernel changes.
