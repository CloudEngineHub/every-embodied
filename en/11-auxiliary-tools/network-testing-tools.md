# Network Testing Tools

`nethogs` groups network traffic by process, which is useful when model or dataset downloads are unexpectedly slow. Install and run it with elevated privileges:

```bash
sudo apt update
sudo apt install -y nethogs
sudo nethogs
```

![image-20251026115247175](../../11-其他辅助工具/assets/image-20251026115247175.png)

Use `ip -brief address` to identify interfaces, `ping` to check reachability, and `curl -I URL` to verify an HTTP endpoint. `ss -lntp` shows listening TCP ports, while `nethogs <interface>` helps confirm whether a training process, package manager, or download client is consuming bandwidth. Record the selected interface and test URL when reporting throughput so that results can be reproduced.
