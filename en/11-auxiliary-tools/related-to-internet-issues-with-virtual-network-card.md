# Set a Windows Virtual Adapter to a Private Network

When a proxy-created virtual adapter is classified as a public network, Windows Firewall rules may block local-network traffic. Change only the target virtual adapter to the private profile.

## Update the network profile

1. Press `Win+R`, enter `secpol.msc`, and open **Local Security Policy**.
2. Open **Network List Manager Policies**.
3. Select the adapter created by the proxy application, such as Mihomo.
4. Set **Location type** to **Private**, apply the change, and reconnect the proxy.

Confirm the adapter by both name and description so that the physical network interface is not changed accidentally.

## Verify the result

Run the following commands in PowerShell:

```powershell
Get-NetConnectionProfile
Test-NetConnection github.com -Port 443
```

The target adapter should report `Private`, and the HTTPS connectivity test should succeed. On Windows editions without `secpol.msc`, identify the correct interface with `Get-NetConnectionProfile` and use the administrator-only `Set-NetConnectionProfile` command to change its category.

Source discussion: <https://linux.do/t/topic/276425>
