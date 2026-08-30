# Linux Network Bandwidth and Connectivity Tests

Network diagnosis should separate process bandwidth, host-to-host throughput, and external connectivity. This page uses `nethogs`, `iperf3`, and `curl` for those three questions.

## Inspect Per-Process Bandwidth

```bash
sudo apt-get update
sudo apt-get install -y nethogs
sudo nethogs
```

The interface shows current upload and download rates grouped by process. Press `q` to exit.

![nethogs process bandwidth view](../../11-其他辅助工具/assets/image-20251026115247175.png)

## Test Throughput Between Two Machines

Install `iperf3` on both machines first:

```bash
sudo apt-get install -y iperf3
```

Run on the server:

```bash
iperf3 -s
```

Run on the client:

```bash
iperf3 -c <SERVER_IP> -P 4
```

## Verify Connectivity and Diagnose Failures

Install `curl` first if it is not already available:

```bash
sudo apt-get install -y curl
```

```bash
curl -I --connect-timeout 10 https://github.com
ip route
```

If a domain fails while direct IP connectivity works, inspect DNS configuration. If `iperf3` cannot connect, check the server port and firewall. If one process consumes most of the available bandwidth, return to `nethogs` before changing download concurrency.

`iperf3` measures transfer capacity between two machines on the local network, `curl` checks connectivity and the response from a selected website, and `nethogs` shows the bandwidth currently used by each local process. These results answer different questions and are not interchangeable.
