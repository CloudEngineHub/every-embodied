# Embodied AI Interview Preparation and Review Template

Use this template to turn project experience into verifiable technical answers and to review questions after an interview. Copy it into a personal note before filling it in. Do not add names, contact details, confidential company information, or unpublished data to the public textbook.

## 1. Project Brief Before the Interview

Prepare one brief for each major project and keep it within one page.

| Item | What to record |
| --- | --- |
| Task | What the robot must accomplish and in which environment |
| Observations | Images, depth, point clouds, joint states, language instructions, and other inputs |
| Actions | Joint positions, end-effector poses, velocity, torque, or discrete actions |
| Data | Scale, collection method, and train/evaluation split |
| Method | Policy architecture, loss functions, planner, or controller |
| Training | Hardware, steps, batch size, learning rate, and elapsed time |
| Evaluation | Tasks, episodes per task, success criteria, and fixed seeds |
| Results | Success rate, latency, throughput, memory use, and common failure stage |
| Contribution | Modules implemented, issues fixed, and traceable contributions |

Structure the spoken project summary as task, constraints, method, results, and lessons:

1. **Task:** State what the robot must do in one sentence.
2. **Constraints:** Name the relevant hardware, data, time, or deployment limits.
3. **Method:** Explain only the design choices that directly affect the result.
4. **Results:** Give numbers and observations under a fixed evaluation protocol.
5. **Lessons:** Identify the main failure mode and the highest-value next improvement.

## 2. Technical Question Log

Record interview questions in the following table. For an unanswered question, also record the reasoning attempted during the interview. This helps separate gaps in concepts, mathematics, implementation, and experimental practice.

| Question | Initial answer | Evidence or equation | Gap | Revised answer |
| --- | --- | --- | --- | --- |
| Example: Why does behavioral cloning suffer from distribution shift? |  |  |  |  |
| Example: How should physical grasp success be defined? |  |  |  |  |
| Example: How are images, states, and actions aligned in time? |  |  |  |  |

## 3. Experimental Question Checklist

When discussing training, deployment, or evaluation, cover the following information when relevant:

- tensor shapes, coordinate frames, action units, and control frequency;
- episode boundaries, timestamps, normalization statistics, and invalid trajectory handling;
- whether training and inference use the same observation and action contract;
- whether success checks contact, lift, transport, placement, release, and other physical stages;
- random seeds, task count, episodes per task, and aggregation method;
- model latency, control period, memory use, and failure recovery behavior.

## 4. Review After the Interview

Complete three actions soon after the interview:

1. Classify each new question under robotics, perception, planning, control, learning, or deployment.
2. Attach a textbook section, paper, source file, or experiment record to every important conclusion.
3. Select the weakest answer and write a minimal validation program or derivation with an expected output.

The review is complete when the input, processing, output, assumptions, and verification method can be explained without reopening the original answer.
