# README 06: OpenClaw Deployment, Takeover, and Board Control

After completing this chapter, readers should be able to accomplish four things on MagicBox. First, verify whether `magicclaw.service` and `magicclaw-buttons.service` are functioning properly. Second, understand what the left, middle, and right physical buttons represent, as well as which actions they can be remapped to. Third, know which `magicboxctl` commands involve pure local hardware control, and which commands require the ROS voice link to be activated first. Fourth, understand what the OpenClaw backend and the Lite frontend provided in this tutorial are responsible for, and how the web frontend should be accessed.

## 1. Acceptance objectives after completing this chapter

This chapter is not just about opening a web page, but requires the following items to be established simultaneously.

- `magicclaw.service` is located within `active (running)`, and `curl http://127.0.0.1:18789/health` can return a healthy status.
- `magicclaw-buttons.service` is situated in `active (running)`, and the three physical buttons can now be re-mapped by MagicClaw.
- `magicboxctl buttons show`, `magicboxctl voice ...`, and `magicboxctl servo ...` can all be correctly understood; commands that require pre-service are no longer treated as "independent execution at any time".
- Readers know that OpenClaw has a backend related to browser control, but the currently stable web entry point should mainly rely on the Lite frontend provided in this tutorial.

## II. Which execution mode is used in this chapter

After MagicBox is integrated with OpenClaw, there are actually two different operating modes.

The first set is the original factory mode, that is, `magicbox-start` mode. It still has `/userdata/magicbox/launch/start.py` handling the key input logic. The left button is fixed to activate the dual-eye mode, the middle button is fixed to activate the gesture mode, and the right button is fixed to activate the voice mode. It is closer to the original factory demonstration experience, with the lighting, sound cues, and key entries following the original design. However, it is not suitable for tutorial scenarios where key positions need to be dynamically changed in the frontend chat box.

The second set is the mode used in this chapter, which is MagicClaw taking over the button mode. In this mode, `magicclaw-buttons.service` is responsible for monitoring three physical buttons, `magicboxctl buttons set ...` is responsible for writing button mappings, and `magicboxctl buttons invoke ...` is responsible for triggering button actions. This approach is more suitable for course demonstrations, front-end dialogue control, and board-side capability orchestration.

These two modes cannot be used as key input simultaneously. The reason is straightforward: the original `magicbox-start` itself listens for keys, and if `magicclaw-buttons.service` also listens at the same time, it will cause a single key press to trigger both modes. Therefore, this chapter defaults to using MagicClaw to handle the key mode, and `magicbox-start` is not permanently enabled as a key input.

## III. Confirm the environment before starting

All commands in this chapter are executed by default on the development board terminal, unless it is explicitly stated to be executed on a Windows host. The following conditions should be met first.

- The development board is connected via Ethernet cable, and the IP address at the board end remains `192.168.127.10`.
- You can log in to the development board using `ssh sunrise@192.168.127.10`.
- The OpenClaw / MagicClaw bundle has been installed in `/userdata/magicclaw`.
- Currently, the MagicClaw takeover button mode is used at the board end, rather than the original factory button mode.

First, execute the following set of check commands in the terminal of the development board:

```bash
ls -ld /userdata/magicclaw
ls -l /userdata/magicclaw/runtime/bin/magicboxctl
ls -l /etc/systemd/system/magicclaw.service
ls -l /etc/systemd/system/magicclaw-buttons.service
```

If these paths exist, it indicates that the installation result on the board end is complete. All subsequent commands in this chapter will be based on these paths.

## IV. Check the service status before formal manipulation

This set of commands can be executed in any directory without the need for the `source` ROS environment, as it checks the `systemd` service itself.

```bash
sudo systemctl status magicclaw.service --no-pager
sudo systemctl status magicclaw-buttons.service --no-pager
curl http://127.0.0.1:18789/health
```

Seeing these results indicates that the browser integration and button control can continue.

- `magicclaw.service` displays `active (running)`.
- `magicclaw-buttons.service` displays `active (running)`.
- The health check returns results similar to `{"ok":true,"status":"live"}`.

If this step fails, do not open the browser first. Since both the old dashboard and the Lite page are only control interfaces, when the gateway service is not working, the web pages will not function properly.

## 5. How to view the left, middle, and right physical buttons respectively

In this chapter, the following direction definitions are used uniformly to avoid ambiguity as to whether "left and right ends are viewed from a person or from a board".

- `left`: Left button when the device faces the user.
- `middle`: Middle button.
- `right`: Right button when the device faces the user.

The GPIO definitions monitored by the current key daemon process are as follows.

| Tutorial Term | GPIO(BCM) | Description |
| --- | --- | --- |
| `left` | 22 | Left button when the device faces the user |
| `middle` | 16 | Middle button |
| `right` | 26 | Right button when the device faces the user |

If you want to view the current mapping relationship, you can directly execute:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl buttons show
```

After successful execution, a JSON string should be returned, for example:

```json
{
  "left": "stereo",
  "middle": "gesture",
  "right": "voice_chat"
}
```

This indicates which action each of the left, middle, and right physical keys is currently bound to.

## What do the six or seven button actions mean respectively?

There are a total of seven actions that can be written into the key mapping currently: `stereo`, `gesture`, `voice_chat`, `voice_asr`, `yolo`, `stop_all`, and `off`. These actions are not abstract names, but rather execution branches that have already been implemented in `magicboxctl`.

### 1. `stereo`

`stereo` is used to start the dual-eye Demo. After pressing it, the dual-eye link is activated, the lights turn red, and the active mode is recorded as `stereo`. If the same `stereo` key that is already in active state is pressed again, the current dual-eye mode will be turned off.

### 2. `gesture`

`gesture` is used to start the gesture demo. After pressing it, the light will turn green, and the active mode will be recorded as `gesture`. If the same `gesture` key that is already in active state is pressed again, the current gesture mode will be turned off.

### 3. `voice_chat`

`voice_chat` is used to start the complete local voice conversation link. The current implementation is aligned with the original right-click voice mode: first ensure that the `audio_io` voice base exists, then activate `qwen_llm` separately, without having to activate the second `audio_io` repeatedly. After pressing it, the light turns blue, and the active mode is written as `voice_chat`. If the same `voice_chat` key that is already in the active state is pressed again, the current voice conversation mode will be turned off.

### 4. `voice_asr`

`voice_asr` only activates the speech recognition and announcement module, that is, the `audio_io` link. It does not activate the entire local LLM conversation. It sets the lighting to blue and writes the activity mode as `voice_asr`. In this chapter, it is not described as "pressing the same button again is equivalent to turning off". If you want to turn it off explicitly, executing `magicboxctl voice asr stop` directly is more reliable.

### 5. `yolo`

`yolo` is used to start the YOLO Demo. After pressing it, the light will turn magenta, and the active mode will be recorded as `yolo`. If the same key is pressed again, it will switch to normal mode to turn it off.

### 6. `stop_all`

`stop_all` serves to stop all current modes. It clears the active status and attempts to turn off the lights. This action is suitable as a dedicated key mapping for "emergency stop" or "return to idle state".

### 7. `off`

`off` does not mean "shut down", but "keep this key, but no mode is activated when pressed". If a key is temporarily not intended to have a function assigned, it can be mapped to `off`.

The summary table is provided below.

| Action | Actual Function | Light | Written as active? | Turns off when same key is pressed again? |
| --- | --- | --- | --- | --- |
| `stereo` | Starts the dual-eye Demo | Red | Yes | Yes |
| `gesture` | Starts the gesture Demo | Green | Yes | Yes |
| `voice_chat` | Ensure `audio_io` exists first, then start `qwen_llm` | Blue | Yes | Yes |
| `voice_asr` | Only starts `audio_io` | Cyan | Yes | Not hardcoded as "Yes" in the tutorial |
| `yolo` | Starts the YOLO Demo | Magenta | Yes | Yes |
| `stop_all` | Stops all modes | Lights off | Clears | Not applicable |
| `off` | No action executed | Unchanged | No | Not applicable |

## VII. How to view, modify, and trigger the three buttons

This set of commands can be executed in any directory by default, without relying on a ROS environment.

First, check the current mapping:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl buttons show
```

Write in another set of default mappings that closely mimic the original factory experience:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl buttons set left stereo
sudo /userdata/magicclaw/runtime/bin/magicboxctl buttons set middle gesture
sudo /userdata/magicclaw/runtime/bin/magicboxctl buttons set right voice_chat
```

After successful execution, the terminal usually returns:

- `left mapped to stereo`
- `middle mapped to gesture`
- `right mapped to voice_chat`

If you want to simulate key presses without pressing physical keys, you can execute:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl buttons invoke right
```

If the current `right` has been mapped to `voice_chat`, the expected result is:

```text
right triggered voice_chat
```

To check if the key daemon service is still running, you can execute:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl buttons daemon status
```

The success criterion for this section is not merely that the command is displayed, but also that it can answer two questions: what actions are associated with the current three physical keys, and whether the result matches the mapping after simulating pressing one of them.

## VIII. Which commands involve pure local hardware control, and which commands require the activation of the link first

The commands `magicboxctl` are not all of the same type.

The first category consists of pure local hardware control commands, such as `led pixel`, `led pattern`, `servo set`, and `servo greet`. These commands directly invoke local hardware scripts, do not rely on ROS topic subscribers, and do not require the voice link to be started first.

The second type consists of commands that require a existing local ROS link, such as `voice say` and `voice prompt`. These two commands do not essentially "play a sentence directly", but instead send a `std_msgs/String` message to a specific ROS topic.

- `voice say` published to `/tts_text`
- `voice prompt` published to `/prompt_text`

If there is no corresponding subscriber currently, they will wait and the following message will appear:

```text
Waiting for at least 1 matching subscription(s)...
```

This is not a crash, but there are no consumers listening to these two topics at the moment.

The third category is mode switching commands, such as `buttons set`, `buttons invoke`, `voice asr start`, and `voice chat start`. These commands activate or deactivate a specific mode link and change the active state at the board.

## 9. How should the voice link be started, and why does `voice say` get stuck?

If the goal is simply to bring the board into the voice recognition and broadcasting base state, you can execute:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl voice asr start
```

After the execution is successful, check the status again:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl voice status
```

The ideal result should include at least:

```text
voice_asr: running
```

If you want to start a full voice dialogue link instead of just activating `audio_io`, follow these steps:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl voice chat start
```

Check the status again:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl voice status
```

Ideal result:

```text
voice_asr: running
voice_chat: running
```

The current `magicboxctl voice chat start` is already aligned in the original right-click mode: first pull up `audio_io`, then start `qwen_llm`, and use the same ROS middleware configuration as the board node. Therefore, it is more stable than "starting the second `audio_io` repeatedly in one command", and can also avoid audio device conflicts.

It is only meaningful to execute the following text injection commands after the above state already exists:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl voice say "你好，这是板端朗读测试。"
sudo /userdata/magicclaw/runtime/bin/magicboxctl voice prompt "请只回答：测试通过。"
```

Both commands have been verified in real machines on the board. `voice prompt` will send the text to the local `qwen_llm`, which then passes it to `/tts_text` for `audio_io` to report.

## Ten. How the servo commands work, and how to interpret left and right

By default, this section is executed in any directory, and the `source` ROS environment is not required. Currently, the servo control uses local PWM scripts instead of ROS nodes.

`left` and `right` in the tutorial refer to the left and right support angles when the device faces the user. In the current script, the left PWM pin is 32, and the right PWM pin is 33. The safe angle range should be maintained between `0-90`, and `30` is recommended as the midpoint.

For the first test, it is recommended to execute the following set of commands first:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl servo set left 15
sudo /userdata/magicclaw/runtime/bin/magicboxctl servo set right 45
sudo /userdata/magicclaw/runtime/bin/magicboxctl servo greet 1
```

If the operation is successful, you should see the left and right support angles move separately at the board end. Finally, `servo greet 1` will perform a brief greeting motion. This control system has been modified according to the official `/userdata/magicbox/basic_function_demo/servo.py` rudder activation method. It uses PWM start and hold times that are closer to the official examples, and thus has been verified in actual use.

It is also necessary to clarify one implementation boundary: the current `servo set` is not a locking control that "maintains the angle continuously", but rather "generates a action pulse and releases the GPIO after a short period of time". Inside the script, `GPIO.cleanup()` is executed after the action ends. Therefore, if readers expect the device to remain locked at that angle after executing the command, this does not match the current implementation. This must be understood clearly to avoid misinterpreting "no continuous locking" as "the command did not take effect".

If there is no reaction after executing the command, the troubleshooting order should be: first check the permissions, then check if the GPIO is occupied, and finally check the hardware connections. Do not check ROS first, as this part does not rely on ROS at all.

## 11. How to access the web front end

This section first clarifies a potentially confusing issue: OpenClaw comes with the backend related to browser control, but it does not mean that there is always an official front-end page that can be opened remotely in the current environment.

On the current dashboard, two types of web-related entries have been identified:

- `ws://0.0.0.0:18789`: Gateway WebSocket, accessible to the local network.
- `http://127.0.0.1:18791`: Browser control backend interface, listening only on the local machine and requiringBearer Token authentication.

This means that OpenClaw indeed has the backend capabilities for web control, but it does not provide an official front-end page that is "stably available through direct remote access" on your current board. In particular, the address `18791` is essentially a backend interface, not a web UI that can be directly accessed by users. Opening it directly in a browser results in the display of `Unauthorized`, which is a normal outcome consistent with the backend design.

Therefore, the conclusion of this tutorial is very clear:

- If the focus is only on studying the built-in backend of OpenClaw, and attention is given to the two interfaces `18789` and `18791`.
- If a stable web entry point suitable for course screenshots, dialogue control, and local network debugging is desired, the Lite front-end provided in this tutorial should be used.

## XII. Why the old dashboard is not suitable as the current main entry

The old dashboard address is usually written as:

```text
http://192.168.127.10:18789/chat?session=main
```

It can work in some environments, but not in the one you are currently in:

- Direct connection via network cable
- HTTP non-secure context
- Not localhost at the board terminal
- Browser accesses directly from a local network address

In such scenarios, it is easy to list:

```text
device identity required
```

It is not that "the board is not activated," nor that "the port is not forwarded." A more accurate understanding is that the old control page still has the handshake logic related to device identity verification, but the current access environment does not meet its requirements.

Therefore, this tutorial no longer uses the old `18789/chat` page as the main entry. It can be retained as a historical interface and troubleshooting reference, but it is no longer recommended as the daily operation entry point.

## XIII. Why the Lite frontend is the recommended entry point

The currently stable web entry is the Lite frontend provided in this tutorial. The reason is simple: it is designed specifically for the local network setup, tutorial screenshots, and debugging control of the MagicBox board, avoiding the instability issues in the old OpenClaw web bundle regarding device identity verification.

The Lite frontend is more suitable for the following tasks:

- Directly connect to the client gateway in a local network environment.
- Provide conversational control and troubleshooting in conjunction with `magicboxctl`.
- Used for screenshots, demonstrations, and by readers for reproduction.

The usage recommendations for this chapter are thus clear:

- Use the Lite frontend when you need stable, direct, and reproducible web control.
- When studying the built-in backend of OpenClaw, check `18789` and `18791` separately.
- It is not recommended to use the old `18789/chat` as the current main entrance.

## 14. Common Errors and Recovery Methods

### 1. `Waiting for at least 1 matching subscription(s)...`

This usually appears in `voice say` or `voice prompt`. The reason is not that the command syntax is incorrect, but because there is no subscriber listening for `/tts_text` or `/prompt_text` at the moment. The solution is to first execute:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl voice asr start
```

Or:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl voice chat start
```

Then execute the text injection command.

### 2. `device identity required`

If this error occurs on the old dashboard, do not attribute the problem to the board or service itself. First, execute:

```bash
curl http://127.0.0.1:18789/health
```

If the health check is normal, it indicates that the issue is related to the device identity handshake on the old control page, not that the gateway has not started. In this case, you should directly switch to the Lite frontend to continue control.

### 3. The button has been changed, but pressing it has no response.

Check the service first:

```bash
sudo systemctl status magicclaw-buttons.service --no-pager
```

Check the mapping again:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl buttons show
```

Finally, directly simulate key presses:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl buttons invoke left
```

If `invoke` works but the physical buttons do not respond, the issue lies primarily in the button listening layer, rather than the mapping layer.

### 4. No action occurs after the servo command is executed

First, confirm that the execution is as follows:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl servo set left 15
```

Or:

```bash
sudo /userdata/magicclaw/runtime/bin/magicboxctl servo set right 45
```

If there is no action, first check the permissions and GPIO usage, then examine the hardware connections. Do not check ROS first, as the servo commands do not rely on ROS.

## 15. Chapter Acceptance Checklist

After completing this chapter, readers should be able to self-check using the following checklist.

- I now know that the current tutorial uses the MagicClaw takeover key mode instead of the factory key mode.
- I can clearly explain what each of the seven actions `stereo`, `gesture`, `voice_chat`, `voice_asr`, `yolo`, `stop_all`, and `off` represents.
- I can use `buttons show` to view the current mapping, and `buttons set` to modify the left, middle, and right physical keys.
- I understand that `voice say` and `voice prompt` are not independent local reading commands, but rather message injection commands that rely on ROS subscribers.
- I know that `servo set` is a pure local PWM action that does not depend on ROS, and the current implementation does not lock the angle continuously.
- I realize that OpenClaw has a browser control backend, but in this environment, the stable web entry point should be the Lite frontend, rather than the old `18789/chat` page.
