# RDK X5 闪连与 RDK Studio 使用教程

## 一、RDK Studio简介

RDK Studio 用于管理 RDK 系列开发板，覆盖镜像烧录、系统升级、示例运行和板端应用启动。首次使用开发板时，可以通过闪连完成系统写入和网络配置，不必额外准备显示器、键盘和鼠标。

- 管理RDK全系列板卡：RDK Studio完美支持RDK全系列板卡，可以通过一个软件轻松管理RDK X3、RDK X3、RDK X5、RDK Ultra，直观快速的查看到每一个板卡当前的状态

-  一键烧录/升级板卡系统：为解决板卡镜像管理混乱，不用时占空间，用时找不到的问题，RDK Studio一个软件便可以实现对一个板卡的镜像烧录以及系统升级

- 示例程序一键运行：RDK Studio内置地瓜机器人官方的NodeHub，动动手指即可实现官方示例程序的运行展示，不再需要繁杂的步骤就能体验RDK非凡性能

- 板卡内置应用一键启动：RDK Studio可以一键启动板卡内置的应用，如Vscode、Jupyter等，让你专注于算法的开发

## 二、RDK Studio的使用教程

### 1、RDK Studio安装  

打开[地瓜开发者社区首页](https://developer.d-robotics.cc/)，在右上角的软件导航中进入 RDK Studio 详情页。

![地瓜开发者社区中的 RDK Studio 入口](assets/images/rdk-studio-community-entry.png)

> 本章 RDK Studio 操作截图来源于原教程配套的博客园图片，已统一归档到 `assets/images/`。

这里面是关于RDK Studio的一些介绍以及下载界面，我们点击醒目的<font color="red">"下载RDK Studio"</font>即可跳转到我们具体的下载页面，目前RDK Studio暂时仅支持Windows平台，但是预计将在2025年的第一季度陆续推出Mac OS和Linux的版本

![RDK Studio 功能概览](assets/images/rdk-studio-overview.png)

![RDK Studio 下载页面](assets/images/rdk-studio-download-page.png)



下载页提供 User Installer 和 `.zip` 两种形式：前者直接启动安装程序，后者先下载压缩包。浏览器若提示安装包来源风险，应先核对下载域名和文件来源，再选择是否保留。



![浏览器中的 RDK Studio 下载提示](assets/images/rdk-studio-download-warning.png)

接着我们双击下载的exe文件即可自动安装并启动我们的RDK Studio，还会在桌面生成一个快捷方式，RDK Studio的默认安装路径为“C:\Users\\{User}\AppData\Local\rdk_studio”，由于软件不会修改注册表，因此如果我们不想将软件安装至C盘，可以直接将安装目录迁移到我们想要的地方即可

![RDK Studio 安装后的桌面快捷方式](assets/images/rdk-studio-installed.png)

### 2、RDK Studio的使用

打开软件后我们可以看到RDK Studio的界面如下，接着我们分不同情况来介绍如何使用这个软件

![RDK Studio 主界面](assets/images/rdk-studio-home.png)

### 2.1 新板卡烧录

首先我们拿到一个新的RDK X5板卡打开软件点击左边的烧录部分，可以看到我们可以对我们即将烧录的设备进行选择以及连接，我们也可以点击X5后面的<font color="red">使用闪联（TypeC）</font>按钮来了解闪连的使用流程，我们的板卡RDK X5也只有在闪联的状态下才能进行系统的烧录

![RDK Studio 新板卡烧录入口](assets/images/rdk-studio-flash-device.png)

![RDK X5 闪连接口与操作步骤](assets/images/rdk-x5-flash-connect-guide.png)

根据官方的说明，我们的X5上有两个TypeC的口上图右边的口为电源口，左边的口为闪联USB2口，我们首先将SD卡装入RDK X5，接着使用Type-C数据线连接电脑与RDK X5，然后按住开关按钮后给RDK X5通电并且持续按住开关按钮3-5秒即可进入闪连模式，进入闪连模式后系统会自动将X5识别为一个U盘，接着我们回到RDK Studio的烧写界面点击下一步即可进入系统下载选择配置界面，在这个界面里我们可以选择我们要下载的系统版本以及下载路径等

![RDK Studio 系统镜像版本选择](assets/images/rdk-studio-image-version.png)

我们接着选择下载最新的镜像，点击下一步即可开始选择我们要烧录到的路径，我们选择”/dev/sdb“，如果点击下一步发现这里没有显示有设备的情况的话可以点击黄字旁的刷新重新检测，如果刷新之后依旧没有那代表板子没有进入闪连模式，需要回到一开始重新按住按钮上电

![RDK Studio 目标存储设备选择](assets/images/rdk-studio-target-disk.png)

接着我们点击安装后软件便会自动下载并烧录所选择的镜像

![RDK Studio 正在下载并烧录镜像](assets/images/rdk-studio-flashing.png)

![RDK Studio 镜像烧录完成](assets/images/rdk-studio-flash-complete.png)

稍微等待一段时间之后系统即可安装成功！！！

### 2.2 添加现有板卡

完成了上一步烧录系统后或者我们已经有了一块烧录好系统的板子，重新上电后即可在设备管理页面添加我们的设备，RDK一共有三种添加设备的方式，分别为网线连接、闪连和已知IP的IP地址连接，在这里我们可以根据我们的需求选择对应的连接方式

![RDK Studio 添加已有开发板](assets/images/rdk-studio-add-device.png)

刚完成系统烧录的设备可以直接选择闪联连接。进入下一步后，根据实际连接方式选择与 RDK 板卡对应的网络设备。

![RDK Studio 选择开发板网络设备](assets/images/rdk-studio-network-adapter.png)

下一步选择登录账户。日常开发优先使用普通用户；只有执行系统级配置时再使用 Root 管理账户。

![RDK Studio 选择板卡登录账户](assets/images/rdk-studio-login-account.png)

接着我们便进入了连接无线网的界面，我们选择当前环境下的无线网输入密码即可连接，当然也可以选择跳过这一步

![RDK Studio 配置无线网络](assets/images/rdk-studio-wifi.png)

最后我们根据我们的需求添加这个设备名称描述联系方式即可添加这个设备

![RDK Studio 填写设备说明](assets/images/rdk-studio-device-profile.png)

点击确认后即可在我们的设备管理一栏看到我们刚刚添加的设备啦！！！

![RDK Studio 设备添加成功](assets/images/rdk-studio-device-added.png)

### 2.3 设备管理

我们添加了设备之后，我们便可以在设备管理界面看到我们添加的设备，如下图是对当前界面功能的基本介绍

![RDK Studio 设备管理界面](assets/images/rdk-studio-device-management.png)

我们点击添加应用即可看到当前板卡支持的应用列表，可根据自己的需求进行选择下载或使用

![RDK Studio 板端应用列表](assets/images/rdk-studio-application-list.png)

### 2.4一键启动示例应用

在添加我们的设备之后我们便可以打开左边栏的示例程序，打开之后我们可以发现里面的示例可以在下拉框中选择我们要运行的设备名称以及执行对应的操作，点击下载示例或者是对应的应用即可立刻体验预置项目！

![RDK Studio 示例程序启动界面](assets/images/rdk-studio-sample-app.png)

### 3、RDK Studio的卸载

当我们不需要RDK Studio的时候想要卸载的时候可以打开系统设置，选择应用->安装的应用在搜索框中输入”RDK Studio “即可搜索到应用，接着点击右边三个"···"选择卸载即可

![Windows 系统中卸载 RDK Studio](assets/images/rdk-studio-uninstall.png)










