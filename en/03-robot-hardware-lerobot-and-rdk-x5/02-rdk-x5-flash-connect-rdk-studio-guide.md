# RDK Studio Setup and Board Management

## 1. Introduction to RDK Studio

I wonder if everyone has encountered issues when using AI development boards—such as the large space occupied by the official images, making it difficult to save files, or not being able to find them when needed. Also, official sample projects become hard to locate over time. And for the first time connecting to the network on a development board, you need to prepare multiple peripherals like a monitor, keyboard, and mouse... To address these various inconveniences in daily development, D-Robotics robot has innovatively introduced RDK Studio, an official developer software, marking a new era in smart robot development! This software significantly reduces the development barrier for robots, allowing even users with no prior experience to easily get started with RDK series boards! As an innovative application for robot development, it offers all the features you can imagine:

- Manage all RDK board cards: RDK Studio fully supports all RDK board cards. It allows easy management of RDK X3, RDK X3, RDK X5, and RDK Ultra through a single software, and enables intuitive and quick access to the current status of each board card.

- One-click board image burning/upgrading: To address the issues of chaotic board image management, space occupation when not in use, and inability to find the board during use, RDK Studio allows for image burning and system upgrading of a single board with just one software.

- One-click execution of sample program: RDK Studio comes with the official D-Robotics robot NodeHub. Just with a few taps, you can run and showcase the official sample program, allowing you to experience the exceptional performance of RDK without complex steps.

- One-click startup of built-in applications on the board: RDK Studio allows one-click startup of built-in applications on the board, such as Vscode and Jupyter, enabling you to focus on algorithm development.

## II. Usage Guide for RDK Studio

### 1. RDK Studio Installation

After covering so much, let’s officially start the tutorial section on using RDK Studio! First, we can visit the D-Robotics robot developer’s official website.[Sweet potato developer community homepage](https://developer.d-robotics.cc/)) Find the details interface of our RDK Studio in the software section of the navigation bar at the top right corner

![RDK Studio entry in the D-Robotics developer community](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-community-entry.png)

This content provides an introduction to RDK Studio and the download interface. By clicking on the prominent <font color="red"> "Download RDK Studio" </font>, you can navigate to our specific download page. Currently, RDK Studio only supports the Windows platform, but it is expected to release versions for Mac OS and Linux in the first quarter of 2025.

![RDK Studio feature overview](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-overview.png)

![RDK Studio download page](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-download-page.png)



Next, we can simply click on any packaging method to download. The difference between the User Installer and .zip versions is only that one allows direct download of the installation package, while the other downloads the compressed installation package. If you choose to download using the User Installer method, you may encounter the following warning; simply select "Keep it as is".



![RDK Studio download warning in the browser](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-download-warning.png)

Then, double-click the downloaded executable to install and launch RDK Studio. A desktop shortcut is created automatically. The default installation directory is `%LOCALAPPDATA%\rdk_studio`. Because the software does not depend on registry entries, you can move this directory to another drive if required.

![RDK Studio desktop shortcut after installation](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-installed.png)

### 2. Usage of RDK Studio

After opening the software, we can see the interface of RDK Studio as follows. Next, we will explain how to use this software in different scenarios.

![RDK Studio home screen](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-home.png)

### 2.1 New board programming

First, we obtain a new RDK X5 board and open the software. By clicking on the burn-in section on the left, we can select and connect to the device we are about to burn in. We can also click on <font color="red">(https://example.com/ee_keep_0000) behind the X5 to use the TypeC </font> button to learn how to use TypeC. The RDK X5 board can only perform system burn-in when in TypeC mode.

![Device-flashing entry for a new RDK board](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-flash-device.png)

![RDK X5 flash-connect interface and procedure](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-x5-flash-connect-guide.png)

According to the official instructions, there are two Type-C ports on our X5. The port on the right is the power port, and the one on the left is the Thunderbolt USB2 port. First, we insert the SD card into the RDK X5. Then, use a Type-C data cable to connect the computer to the RDK X5. After pressing the switch button, power on the RDK X5. Holding the switch button for 3–5 seconds will enter the Thunderbolt mode. Once in Thunderbolt mode, the system will automatically recognize the X5 as a USB drive. Next, we return to the burning interface of RDK Studio and click "Next" to enter the system download and configuration interface. In this interface, we can select the system version and download path we want to download.

![Selecting the system image version in RDK Studio](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-image-version.png)

Next, we select the latest image to download. Click Next to start selecting the path where we want to burn the firmware. We choose “/dev/sdb”. If no device is displayed here after clicking Next, you can click the refresh button next to the yellow icon to recheck. If nothing appears after refreshing, it means the board has not entered the flash link mode, and you need to return to the beginning and hold the button to power on again.

![Selecting the target storage device in RDK Studio](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-target-disk.png)

Then, after clicking install, the software will automatically download and burn the selected image.

![RDK Studio downloading and flashing the image](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-flashing.png)

![RDK Studio image flashing completed](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-flash-complete.png)

After a short wait, the system can be installed successfully!!!

### 2.2 Adding Existing Boards

After completing the previous step of system burning, or once we have a board with a burned system, we can add our device to the Device Management page after powering it back on. There are three ways to add devices in RDK: network cable connection, flash connection, and connection via known IP address. Here, we can choose the appropriate connection method according to our needs.

![Adding an existing board in RDK Studio](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-add-device.png)

The device with the system just burned into can directly select the FlashLink connection. After clicking Next, we will be asked to choose the network device for our board. In this step, we only need to select the corresponding network device for the RDK board based on our connection method.

![Selecting the board network adapter](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-network-adapter.png)

Next, when we click Next, it will ask us to select our login device. We can choose between a regular user or a Root user based on our needs. Here, I choose the Root user.

![Selecting the board login account](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-login-account.png)

Next, we enter the interface for connecting to the wireless network. We can choose to input the password of the wireless network in the current environment to connect, or we can skip this step.

![Configuring the wireless network](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-wifi.png)

Finally, we add the device name description and contact information according to our needs, and then we can add this device.

![Entering the device profile](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-device-profile.png)

Click Confirm, and you will see the device we just added in the Device Management section!!!

![RDK Studio device added successfully](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-device-added.png)

### 2.3 Device Management

After adding the device, we can see the device we added in the device management interface. The following figure provides a basic introduction to the features of the current interface.

![RDK Studio device-management screen](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-device-management.png)

By clicking Add Application, you can see the list of applications supported by the current board. You can choose to download or use them according to your needs.

![Applications available on the board](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-application-list.png)

### 2.4 Example Application with One-Click Start

After adding our device, we can open the sample program in the left sidebar. Once opened, we can select the device name we want to run from the dropdown menu and perform the corresponding manipulation. Clicking "Download Sample" or the corresponding application allows us to immediately experience the pre-set items!

![Starting a sample application in RDK Studio](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-sample-app.png)

### 3. Uninstalling RDK Studio

When we want to uninstall RDK Studio because we no longer need it, we can open System Settings, select Apps -> Installed Apps, enter "RDK Studio" in the search box to find the app, and then click the three "···" on the right side to choose Uninstall.

![Uninstalling RDK Studio on Windows](../../03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/assets/images/rdk-studio-uninstall.png)
