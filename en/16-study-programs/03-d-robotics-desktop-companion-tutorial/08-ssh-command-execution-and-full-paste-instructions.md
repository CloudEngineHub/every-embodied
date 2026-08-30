# README 08: SSH Command Execution and Whole-Paragraph Paste Instructions

This chapter addresses a very specific but potentially misleading issue for beginners: why the same set of commands may cause problems when pasted as a whole, but work normally when executed one by one. This phenomenon is particularly common in the ROS2 startup commands of Magicbox, as many startup processes include `sudo su -`, `source`, and `ros2 launch`, and these three types of commands are highly sensitive to the current shell environment.

## I. First, state the conclusion

If a set of commands contains `sudo su -`, do not paste it along with all subsequent commands in a whole paragraph without hesitation. There are two safer approaches.

The first approach is to execute it separately:

```bash
sudo su -
```

After confirming that the prompt has been switched to root, paste the subsequent commands as a whole.

The second approach is to completely avoid interactive shell switching, and instead use a `sudo bash -lc '...'` to encapsulate the entire command set within a single root shell. This is usually a better method for tutorial commands that need to be copied and executed remotely repeatedly.

## II. Why the entire text paste fails

The problem lies not in the Magicbox itself, but in the shell switch.

The function of `sudo su -` is not simply to gain higher privileges, but to start a new root login shell. The new shell will reload its environment and take over the input of subsequent commands. The problem is that many Windows terminals, SSH clients, or web terminals send the following commands continuously as soon as a long block of text is pasted. As a result, the subsequent commands may not all be executed in the shell you expect.

There are three most common results.

First type: The first half of the command is executed in the old shell, while the second half is executed in the new shell.
Second type: `source /opt/tros/humble/setup.bash` takes effect in one shell, but `ros2 launch` actually runs in another shell.
Third type: The command itself is correct, but the environment is not preserved fully, so it appears as if "the entire command has no response" or "`ros2` cannot be found".

This is why you see a phenomenon similar to mysticism: executing each item individually works fine, but pasting the entire paragraph fails.

## III. Which commands are most vulnerable to this issue

In this tutorial, the following three types of commands are most susceptible to changes.

The first type are commands that switch the shell, for example:

```bash
sudo su -
sudo -i
```

The second type consists of commands that apply only to the current shell, for example:

```bash
source /opt/tros/humble/setup.bash
source /userdata/magicbox/app/ros_ws/install/local_setup.bash
export ROS_LOG_DIR=/userdata/magicbox/log
```

The third category consists of commands that strongly depend on the results of the first two categories, for example:

```bash
ros2 launch gesture_interaction gesture_interaction.launch.py
ros2 launch hobot_stereonet stereonet_model_web_visual_v2.4_int16.launch.py
```

Therefore, if three steps—"switch shell", "source environment", and "start ROS2"—appear in a single command, it should not be treated as a complete command block that can be pasted verbatim without any conditions.

## IV. Which execution method is recommended in the tutorial

If you are debugging manually, it is recommended to use "two paragraphs".

In the first paragraph, enter root separately:

```bash
sudo su -
```

In the second paragraph, execute the actual startup command. For example, gesture interaction can be written as:

```bash
cd /userdata/magicbox
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=/userdata/magicbox/app/ros_ws/install/qwen_llm/lib:$LD_LIBRARY_PATH
export HOME=/userdata/magicbox
export ROS_LOG_DIR=/userdata/magicbox/log
mkdir -p /userdata/magicbox/log
source /opt/tros/humble/setup.bash
source /userdata/magicbox/app/ros_ws/install/local_setup.bash
ros2 launch gesture_interaction gesture_interaction.launch.py
```

If you need to repeat the same command multiple times, it is recommended to use a "single command per paragraph".

```bash
sudo bash -lc '
cd /userdata/magicbox
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=/userdata/magicbox/app/ros_ws/install/qwen_llm/lib:$LD_LIBRARY_PATH
export HOME=/userdata/magicbox
export ROS_LOG_DIR=/userdata/magicbox/log
mkdir -p /userdata/magicbox/log
source /opt/tros/humble/setup.bash
source /userdata/magicbox/app/ros_ws/install/local_setup.bash
exec ros2 launch gesture_interaction gesture_interaction.launch.py
'
```

The advantage of this command is that all manipulations occur within the same root shell, eliminating the issue of "the environment sourced earlier going to another shell."

## V. What to Look At First If It Has Already Failed

Don't doubt the repository or launch file first. Instead, check whether the current shell actually retains the required environment.

First, check if `ros2` is in the current PATH:

```bash
command -v ros2
```

If there is no output here, it means the current shell has not loaded the complete ROS2 environment.

Check whether the key log directory is writable:

```bash
ls -ld /userdata/magicbox/log /userdata/.roslog
```

If the directory permissions are incorrect, even if the `ros2 launch` command exists, it may exit before starting due to failed creation of the log directory.

## VI. Which chapters are applicable to this rule

This rule applies not only to gesture interactions. Any command in this tutorial with the following structure should be executed according to this chapter first.

```text
sudo / root shell
-> source ROS2 环境
-> source 工作区环境
-> ros2 launch
```

Therefore, the depth perception and gesture interaction in Chapter 4, as well as any subsequent ROS2 applications that require a root environment, should follow the same execution principles.

## VII. What Readers Should Master After This Chapter

After reading this chapter, the reader should be clear about three things.

First, the failure to paste the entire paragraph but success when executing each item individually does not indicate "mystical instability" of the board; it is often merely a problem with shell switching and environment inheritance.
Second, if the commands contain `sudo su -`, `source`, and `ros2 launch`, consider whether they run in the same shell first.
Third, if a repeatable and copyable stable command is needed, prioritize using the single-command encapsulation method such as `sudo bash -lc '...'`.
