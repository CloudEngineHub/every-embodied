# Connect a Bluetooth Adapter to WSL

Install the USB/IP client inside the WSL distribution:

```
sudo apt install linux-tools-generic hwdata

sudo update-alternatives --install /usr/local/bin/usbip usbip /usr/lib/linux-tools/*-generic/usbip 20


```

```
usbipd list
Connected:
BUSID  VID:PID    DEVICE                                                        STATE
1-3    8087:0029  Intel(R) Wireless Bluetooth(R)                                Shared
1-4    1bcf:2a02  Integrated Webcam                                             Not shared
2-2    30fa:1440  USB Input Device                                              Not shared

Persisted:
GUID                                  DEVICE

usbipd attach --wsl --busid 1-3
usbipd: info: Using WSL distribution 'Ubuntu-22.04' to attach; the device will be available in all WSL 2 distributions.
usbipd: info: Loading vhci_hcd module.
usbipd: info: Detected networking mode 'mirrored'.
usbipd: info: Using IP address 127.0.0.1 to reach the host.
WSL usbip: error: Attach Request for 1-3 failed - Device busy (exported)
usbipd: warning: The device appears to be used by Windows; stop the software using the device, or bind the device using the '--force' option.
usbipd: error: Failed to attach device with busid '1-3'.
# If Windows still owns the adapter, disable Bluetooth in Windows, then retry:
usbipd attach --wsl --busid 1-3
usbipd: info: Using WSL distribution 'Ubuntu-22.04' to attach; the device will be available in all WSL 2 distributions.
usbipd: info: Detected networking mode 'mirrored'.
usbipd: info: Using IP address 127.0.0.1 to reach the host.
```

Replace `1-3` with the `BUSID` reported for the Bluetooth adapter on your machine.
