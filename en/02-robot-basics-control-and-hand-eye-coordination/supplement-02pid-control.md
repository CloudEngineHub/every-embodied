# [PID Control Algorithm and Code Implementation]

## Authors
## [Witty Stream Hardware]  [Leng Xiaomo]
### [1412195676@qq.com]

## Date
[2024-12]

## Abstract
- The PID control algorithm is one of the most commonly used control algorithms in industrial automation.
- PID adjusts the control input by calculating the error between the current output and the desired output, and modifying the control input based on the ratio (P), integral (I), and derivative (D) of this error, thereby achieving precise control of the system.
## Working Principle
The PID controller calculates the control input through the following three basic components:
- Ratio (P): Directly scales the error, which can quickly reduce the error but may lead to system instability.
- Integral (I): Integrates the error, which can eliminate the static error of the system but may slow down the system response.
- Derivative (D): Predicts the derivative of the error, which can forecast the trend of error changes, allowing for early adjustment and improving the system's response speed and stability.
- e(t) is the current error, i.e., the difference between the desired output and the actual output.
- K_p is the ratio coefficient.
- K_i is the integral coefficient.
- K_d is the derivative coefficient.
- Implementation Steps

-
Calculation error: The error between the current output and the expected output is calculated.
Ratio term: The ratio term is calculated based on the ratio coefficient and the error.
Integration term: The error is integrated, and the integration term is calculated based on the integration coefficient.
Differentiation term: The derivative of the error is calculated, and the differentiation term is calculated based on the differentiation coefficient.
Calculation control input: The ratio term, integration term, and differentiation term are added to obtain the control input.
Application control input: The control input is applied to the controlled system.

Overall formula: ![alt text](../../02-机器人基础和控制、手眼协调/assets/image.png)

---



### Example Code (Python)
Here is a simple example of a Python implementation of a PID control algorithm:
Describe in detail your research method, experimental design, data collection, and analysis process. Ensure that the method section is clear and accurate so that other researchers can reproduce your experimental results.
#### (In this case, position-based PID is used; incremental PID, etc. can be modified according to the actual situation, and details will be provided below)
```
class PID:
    # pid的初始化赋值
    def __init__(self, Kp, Ki, Kd, setpoint=0, sample_time=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.sample_time = sample_time

        self.prev_error = 0
        self.integral = 0

    # pid的cal_process
    def update(self, measured_value):
        error = self.setpoint - measured_value # 计算误差
        self.integral += error * self.sample_time # 积分
        derivative = (error - self.prev_error) / self. sample_time # 微分

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative # 计算控制输入
        self.prev_error = error # 保存误差

        return output


pid = PID(Kp=1.0, Ki=0.1, Kd=0.01, setpoint=100)
measured_value = 90  # 假设的当前测量值
control_input = pid.update(measured_value)

print(f"Control Input: {control_input}")

```



### Example code (C language)

```
void pid_init(pid_type_def *pid,uint8_t mode,float max_output,float max_iout,const float PID[3])
{

	pid->mode=mode;
	pid->max_output=max_output;
	pid->max_ioutput=max_iout;
	pid->Dbuf[0]=pid->Dbuf[1]=pid->Dbuf[2];
	pid->error[0]=pid->error[1]=pid->error[2]=pid->pout=pid->iout=pid->dout=0.0f;
}

float PID_calc(pid_type_def *pid,float ref,float set,float PID[3])
{
	pid->kp=PID[0];
	pid->ki=PID[1];
	pid->kd=PID[2];

	pid->error[2]=pid->error[1];
	pid->error[1]=pid->error[0];

	pid->set=set;
	pid->cur=ref;

	pid->error[0]=set-ref;

	if (pid->mode ==PID_POSITION)
	{
		pid->pout=pid->kp*pid->error[0];
		pid->iout+=pid->ki*pid->error[0];
		pid->Dbuf[2]=pid->Dbuf[1];
		pid->Dbuf[1]=pid->Dbuf[0];
		pid->Dbuf[0]=pid->error[0]-pid->error[1];
		if (pid->iout>pid->max_ioutput)
		{
			pid->iout=pid->max_ioutput;
		}
		else if (pid->iout<-pid->max_ioutput)
		{
			pid->iout=-pid->max_ioutput;
		}
		if (pid->iout*pid->error[0]<0)
		{
			pid->iout=0;
		}
		if (pid->cur>0.707*pid->set || pid->cur<0.707*pid->set)
		{
				pid->dout=pid->kd*pid->Dbuf[0];
		}
		else
		{
					pid->dout=0;
		}
		pid->output=pid->pout+pid->iout+pid->dout;
		if (pid->output>pid->max_output)
		{
			pid->output=pid->max_output;
		}
	}

	else if (pid->mode==PID_DELTA)
	{
		pid->pout=pid->kp*(pid->error[0]-pid->error[1]);
		pid->iout=pid->ki*pid->error[0];
		if (pid->iout>pid->max_ioutput)
		{
			pid->iout=pid->max_ioutput;
		}
		else if (pid->iout<-pid->max_ioutput)
		{
			pid->iout=-pid->max_ioutput;
		}
		if (pid->iout*pid->error[0]<0)
		{
			pid->iout=0;
		}
		pid->Dbuf[2]=pid->Dbuf[1];
		pid->Dbuf[1]=pid->Dbuf[0];
		pid->Dbuf[0]=pid->error[0]-2.0f*pid->error[1]+pid->error[2];

		if (pid->cur>0.707*pid->set || pid->cur<0.707*pid->set)
		{
				pid->dout=pid->kd*pid->Dbuf[0];
		}
		else
		{
					pid->dout=0;
		}

		pid->output+=pid->pout+pid->iout+pid->dout;

		if (pid->output>pid->max_output)
		{
			pid->output=pid->max_output;
		}


	}

	return pid->output;
}


```
## Experience Sharing
- Generally, we evaluate the effectiveness of the PID algorithm based on the stability of the waveform and the system response.
- Usually, we can start with a large proportional coefficient and gradually reduce it until the system response becomes smooth.
- The integral term helps eliminate the static error of the system, but its introduction slows down the system response.
- The derivative term predicts the trend of error changes, allowing for early adjustments and improving the system's response speed and stability.


## Example Analysis (MatLab-Simulink)
- ## This is the simulation diagram and its effect diagram in Matlab Simulink when P is 10, I is 0.001, D is 0.01, and the sampling period T is 200.
![alt text](../../02-机器人基础和控制、手眼协调/assets/image-1.png)
- It is shown that a steady-state error exists. In this case, i should be increased to eliminate the steady-state error.
![alt text](../../02-机器人基础和控制、手眼协调/assets/image-2.png)


-  ## After increasing the value of I, overshoot was observed; planning to increase D.
![alt text](../../02-机器人基础和控制、手眼协调/assets/image-3.png)

- ## The best results are as follows
![alt text](../../02-机器人基础和控制、手眼协调/assets/image-4.png)

## PID Tuning for Optimization and Other Scenarios



### There is a mnemonic to share with you all
```
            参数整定找最佳， 从小到大顺序查。

            先是比例后积分， 最后再把微分加。

            曲线振荡很频繁， 比例度盘要放大。

            曲线漂浮绕大弯， 比例度盘往小扳。

            曲线偏离回复慢， 积分时间往下降。

            曲线波动周期长， 积分时间再加长。

            曲线振荡频率快， 先把微分降下来。

            动差大来波动慢， 微分时间应加长。

            理想曲线两个波， 前高后低四比一。

            一看二调多分析， 调节质量不会低。
```


### Additional

- ### Generally, position PID control algorithms can stabilize most control systems. However, depending on the application scenario, different optimization algorithms are also used.

- ### 1. Integral Separation PID

![alt text](../../02-机器人基础和控制、手眼协调/assets/image-5.png)

- ### 2. PID Control for Variable-Speed Integration

![alt text](../../02-机器人基础和控制、手眼协调/assets/image-6.png)

- ### 3. Incomplete Differential PID Control

![alt text](../../02-机器人基础和控制、手眼协调/assets/image-7.png)
