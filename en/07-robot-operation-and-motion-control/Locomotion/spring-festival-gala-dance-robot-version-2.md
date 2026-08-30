# Spring Festival Gala Dance Robot: Follow-Up Learning Path

This page keeps the legacy filename so that existing course links remain valid. The complete environment setup, motion processing, training commands, and result checks are maintained in [Spring Festival Gala Dance Robot Reproduction](./01-spring-festival-gala-dance-robot-recreated.md).

## Learning Sequence

1. Load the robot model, verify the joint mapping, and complete the minimal motion replay in the main tutorial.
2. Enter the `video2robot` directory and extract human pose and motion sequences from a reference video.
3. Retarget the motion to the robot while checking joint limits, foot contacts, and the center-of-mass trajectory.
4. Export a replay video and compare timing, range of motion, and stability with the reference.

The upstream whole-body tracking repository can be cloned with:

```bash
git clone https://github.com/HybridRobotics/whole_body_tracking.git
```

## Reproduction Record

Record the motion source, robot model, control frequency, source frame rate, and replay output for each experiment. When changing robot models, verify joint order, degrees of freedom, and limits before reusing a motion array.

## Completion Criteria

The robot should replay the complete sequence without exceeding joint limits or showing persistent foot penetration. If the motion jitters, first compare the input frame rate with the control frequency, then inspect interpolation after retargeting.
