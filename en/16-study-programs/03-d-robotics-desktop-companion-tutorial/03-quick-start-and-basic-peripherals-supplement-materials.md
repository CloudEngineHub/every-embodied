# README 03 Supplementary Materials: Basic Peripherals Script and Button Detection Source Code

This document is a supplementary reading material for [README_03_Quick Start and Basic Peripherals.md](03-quick-start-and-basic-peripherals.md). The main content has covered the essential operations such as startup, button experiences, and basic peripheral testing. This supplementary material highlights several key underlying scripts that are worth retaining, helping readers progress from being able to operate and run things to being able to understand and modify them.

It should be noted that the code snippets listed below come from the board directory `/userdata/magicbox/basic_function_demo/`. They are not the complete implementation of the product logic, but rather the minimum examples retained by the official image for teaching and hardware verification. Therefore, the focus when reading them is not on achieving high code complexity, but on understanding how each hardware device is accessed, initialized, and controlled on the board.

## 1. Actuator script: The minimal entry point for understanding PWM control

`servo.py` is the easiest-to-understand script segment, and it is also ideal for beginners. It directly uses `Hobot.GPIO` to activate two pins that support PWM, and outputs a duty cycle at a frequency of 50Hz, thereby driving the two servos. The core code is as follows.

```python
output_pin = 32
output1_pin = 33

GPIO.setmode(GPIO.BOARD)
p = GPIO.PWM(output_pin, 50)
p1 = GPIO.PWM(output1_pin, 50)
p.ChangeDutyCycle(5)
p.start(5)
p1.ChangeDutyCycle(11)
p1.start(11)
```

This section of code conveys three very important messages. First, the servo control on the Magicbox is not a "black-box capability"; the underlying mechanism is still standard PWM output. Second, the official examples use the `BOARD` numbering scheme. Therefore, when modifying scripts, it is essential to confirm that the board pin numbers are used instead of BCM numbers. Third, the servo movement is ultimately determined by the duty cycle. Whether it is gesture interaction or large model function calls, as long as they involve physical motion, they all ultimately rely on this control logic.

The value of this script lies not in "ending after one execution," but in establishing a minimal cognitive model for all subsequent action-related sections: at the top is gesture, voice, or LLM, and at the bottom is always PWM control.

## II. LED script: Understanding why WS2812B uses SPI instead of ordinary GPIO

`ws2812b.py` demonstrates another very typical board-side control method. WS2812B LED beads have strict timing requirements. The official script does not use simple high-low level flips to directly control the timing; instead, each RGB data is encoded into a SPI bit stream and sent via `spidev`. The core structure is as follows.

```python
WS2812B_BIT_PATTERNS = {
    0: [1, 0, 0],
    1: [1, 1, 0],
}

self.spi = spidev.SpiDev()
self.spi.open(1, 0)
self.spi.max_speed_hz = spi_speed
self.spi.mode = 0
```

This indicates that the Magicbox lighting control is not simply "setting a color value," but rather sending the encoded timing data via SPI to the light strip as a whole. The script also includes a detail with educational significance: WS2812B uses the `GRB` sequence, instead of the `RGB` sequence that many beginners intuitively assume. It is precisely because these underlying details are easily overlooked that they deserve to be explained separately in tutorials.

If you plan to integrate OpenClaw, nanobot, or custom desktop pets into the Magicbox later, this script serves as the most natural "lighting bridge layer". The remote assistant does not need to understand the SPI protocol; it only needs to call a local script interface. This script is responsible for converting colors and timing into control commands that the device can execute.

## III. IMU Script: Understanding Why the Attitude Sensor is Read via I2C

`imu.py` shows another typical path for the sensor reading. The script opens I2C bus 5 through `smbus2.SMBus(5)`, communicates with `ICM20948` with address `0x68`, then reads the accelerometer and gyroscope registers, and converts the raw values into `m/s²` and `rad/s`. The core logic is as follows.

```python
ICM20948_ADDRESS = 0x68
bus = smbus2.SMBus(5)

ax = read_i2c_word(ACCEL_XOUT_H) * accel_scale
gy = read_i2c_word(GYRO_XOUT_H + 2) * gyro_scale
```

The significance of this section is that it helps readers understand the differences between peripherals such as "visual, motor, and sensor" devices. A servo uses PWM, a light strip uses SPI, and an IMU uses I2C. The reason why Magicbox is suitable as a multimodal platform is not that the official documentation encapsulates all capabilities into a unified API, but because the board itself provides these common hardware interfaces. Once you understand this, whether you integrate OpenClaw or write your own pet robot state machine, you won't confuse "perception" and "execution" with the same control method.

## IV. Button script: It can describe the hardware connection, but cannot replace the system startup entry

`button.py` is easily misread. Many people, seeing that it monitors three GPIO pins, assume it is the official implementation of the "three-key activation of three functions" system. In reality, this script only performs a more basic task: waiting for the falling edge events of the three buttons, and then printing `button1 OK`, `button2 OK`, and `button3 OK` in the terminal. The core code is as follows.

```python
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin1, GPIO.IN)
GPIO.setup(pin2, GPIO.IN)
GPIO.setup(pin3, GPIO.IN)

GPIO.wait_for_edge(self.pin1, GPIO.FALLING)
print("button1 OK")
```

This means it is more like a detection script to check whether the "button hardware is powered on," rather than a product-level scheduler. For this reason, in the tutorial, this script should be positioned as "auxiliary explanation of hardware pathways," not as "the source entry for system preset applications." The former answers the question of "whether the button is detected," while the latter explains "why the dual-eye, gesture, or voice function is activated after the button is pressed." These two issues must be addressed separately; otherwise, readers might mistake the hardware testing script for the overall system control framework.

## 5. How should this supplementary material be used?

A more efficient way of using them is not to mechanically memorize the four scripts, but to regard them as templates for four underlying interfaces. After studying the main text, when reviewing this supplementary material, readers should be able to understand that when upper-level applications require actions, it usually falls on PWM; when light feedback is needed, it usually falls on SPI; when姿态 data is required, it usually falls on I2C; when physical buttons are needed as startup or switch signals, the first thing to address is GPIO event listening.

Once this underlying understanding is established, Magicbox is no longer just a "running demo development board," but a hardware system that can be further developed, connected, and reconfigured. The subsequent sections on dual eyes, gestures, voice, and OpenClaw will all be based on this level of understanding.
