**Special attention should be paid to the last step of configuring a domestic proxy, otherwise packages cannot be downloaded in China.**

------

### Option 1: Install using Conda (Recommended; no root privileges required)

Since you often use the Python environment, installing Go using `conda` is the simplest and cleanest way, and it doesn't require administrator (sudo) privileges.

Bash

```
# 在你的基础环境或特定环境中安装
conda install -c conda-forge go
```

- **Advantages**: One-click installation, newer version, and does not contaminate the system environment.
- **Disadvantages**: Only effective in the current Conda environment.

------

### Option 2: Standard installation of Linux server (official binary)

If you need global installation (and have `sudo` permissions), or don't want to rely on Conda, this is the standard approach.

1. **Download installation package** (using Linux x86-64 as an example, version 1.23.4):

   Bash

   ```
   wget https://go.dev/dl/go1.23.4.linux-amd64.tar.gz
   # 如果 go.dev 访问慢，可以使用国内源：
   # wget https://mirrors.ustc.edu.cn/golang/go1.23.4.linux-amd64.tar.gz
   ```

2. **Unzip to the installation directory** (usually `/usr/local`):

   Bash

   ```
   sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.23.4.linux-amd64.tar.gz
   ```

3. Configure environment variables:

    The Go binary directory needs to be added to the PATH.

   Bash

   ```
   # 编辑 ~/.bashrc
   echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
   # 同时也把 Go 安装第三方工具的目录加进去（比如安装 s5cmd 后会在 $HOME/go/bin）
   echo 'export PATH=$PATH:$(go env GOPATH)/bin' >> ~/.bashrc

   # 刷新配置
   source ~/.bashrc
   ```

------

### Option Three: Windows / macOS Installation

- **Windows**: Directly go to the official website to download the `.msi` installation package, double-click it to run, and select "Next" until completion. It will automatically configure the environment variables.
- **macOS**: If Homebrew is installed, directly run `brew install go`.

------

### 🚨 Key Step: Configure the Domestic Acceleration Proxy (GOPROXY)

This step is very important!

If not configured, running `go install github.com/...` in China will almost 100% fail due to a timeout, as the default Go source is blocked.

Please execute the following command in the terminal to enable the domestic proxy (Qiniu Cloud or Alibaba Cloud source):

Bash

```
# 启用 Go Modules 功能
go env -w GO111MODULE=on

# 配置 GOPROXY 代理 (推荐使用七牛云的 goproxy.cn，速度很稳)
go env -w GOPROXY=https://goproxy.cn,direct
```

------

### Verify Success

After installation and configuration, run the following command to test:

1. **Check version:**

   Bash

   ```
   go version
   # 输出示例: go version go1.23.4 linux/amd64
   ```

2. **Now you can install s5cmd:**

   Bash

   ```
   go install github.com/peak/s5cmd/v2@latest
   ```

   *(After installation, `s5cmd` may be located under `~/go/bin/`. If PATH is configured earlier, simply typing `s5cmd` will work)*





# Notes: Other download methods



This is a standard **AWS S3** bucket path. To download the content under this path, traditional "API Key" is usually not required; instead, the **AWS CLI** command-line tool can be used.

For your three questions, the answers are as follows:

1. **How to download:** Use AWS CLI or S3-compatible tools (such as `s5cmd`).
2. **Is the API required?** If it is a public dataset (OpenPI appears to be an open-source project), usually **no** account or key is needed (use `--no-sign-request` parameter). If it is private, AWS credentials (Access Key) need to be configured.
3. **Can it be accelerated with multiple threads?** **Yes.** AWS CLI has built-in concurrency capabilities, but using `s5cmd` tool usually allows full bandwidth utilization and faster speed.

Here are the specific download options, sorted by recommendation level:

------

## Method 1: Using `s5cmd` (Recommended, fastest)

If you are looking for **ultra-fast multi-thread download speed**, I strongly recommend [s5cmd](https://github.com/peak/s5cmd). It is written in Go and has excellent concurrency performance, much faster than the official CLI.

1. **Install s5cmd:**

  - **Linux (via go):** `go install github.com/peak/s5cmd/v2@latest`
   - **Linux (direct download):** Download the binary file from the release page and extract it.
   - **Conda:** `conda install -c conda-forge s5cmd`

2. Execute the download (assuming it is a public bucket):

--no-sign-request indicates anonymous access without the need for login.

   Bash

   ```
   s5cmd --no-sign-request cp --concurrency 50 "s3://openpi-assets-simeval/pi0_fast_droid_jointpos/*" ./local_folder/
   ```

  - `--concurrency 50`: Enable 50 threads for concurrency (you can adjust it according to your bandwidth).

------

### Method 2: Using the official AWS CLI (standard practice)

This is the most general method.

1. **Install AWS CLI:**

   Bash

   ```
   pip install awscli
   # 或者使用 conda
   conda install -c conda-forge awscli
   ```

2. Execute the download:

Since this is a directory, the `sync` command or `cp --recursive` should be used.

    Case A: If it is a public dataset (Public Bucket)

    Typically, the dataset of open-source projects (such as OpenPI) is publicly available without the need for an AWS account login.

   Bash

   ```
   aws s3 sync s3://openpi-assets-simeval/pi0_fast_droid_jointpos ./local_target_folder --no-sign-request
   ```

- Note: `--no-sign-request` is crucial; it tells AWS, "I don't have an account, let me try anonymous downloading."

    Case B: If it is a private dataset

    If the above command reports an error "Access Denied", it means you need permissions. You need to:

   1. Register a AWS account.
   2. Obtain the Access Key ID and Secret Access Key.
   3. Run `aws configure` and enter the key.
   4. Remove the `--no-sign-request` parameter and run again.

3. How to speed up AWS CLI:

   AWS CLI is multi-threaded by default, but you can optimize performance by modifying the configuration:

   Bash

   ```
   aws configure set default.s3.max_concurrent_requests 20
   aws configure set default.s3.max_queue_size 10000
   ```

------

### Method 3: Using Python (`boto3`)

If you want to load directly in the code instead of downloading to the hard drive, you can use the `boto3` library in Python.

Python

```
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# 配置匿名访问 (对应 --no-sign-request)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

bucket_name = 'openpi-assets-simeval'
prefix = 'pi0_fast_droid_jointpos'

# 列出文件
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

for page in pages:
    if 'Contents' in page:
        for obj in page['Contents']:
            print(f"Found: {obj['Key']}")
            # 在这里添加下载逻辑: s3.download_file(...)
```

### Summary Recommendations

- **Preferred:** Try `aws s3 sync ... --no-sign-request` in **Method 2**. This is the fastest way to verify whether permissions are required.
- **If the file is large:** Be sure to use **Method 1 (`s5cmd`)**. In cases of large files and many small files, its speed is usually several times that of the official CLI.

**Do you need me to write a Python script that automatically checks if the Bucket is public and selects the fastest way to download it?**