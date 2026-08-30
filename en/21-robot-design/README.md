# robotic arm and robot design

This directory contains tutorials related to robot mechanical structures, code modeling, CAD generation, robot description files, and learning-driven robot design. It is recommended that you start with parametric modeling before moving on to workflows such as Text-to-CAD, ForgeCAD, URDF/SDF for robot projects, and value-gradient robot design.

## Tutorial List

- [Build123d Code Modeling Introduction](01-build123d-code-modeling-introduction/README.md) - Generate STEP, STL, and preview images using Python/build123d to understand the basic syntax of CAD code.
- [Text-to-CAD Engineering Modeling Introduction](02-text-to-cad-engineering-modeling-introduction/README.md) - Use the Code Agent, CAD skill, and CAD Explorer to process source files, derived files, and perform geometric checks.
- [ForgeCAD Official 3D Printer, Keyboard, and Dexterous Hand Reproduction](03-forgecad-visual-reverse-engineering-introduction/README.md) - Align with the `3dprinter-gpt52codex` benchmark of ForgeCAD public kit, save the official GIF, and replicate the parametric assembly of the 3D printer, video keyboard, and movable dexterous hand, along with rendering GIFs and exporting to STEP/STL/3MF.
- [Shape Your Body Value Gradient Robot Design Guide](04-shape-your-body-value-gradient-robot-design-introduction/README.md) - Explain how to train multi-body policy/value functions and optimize continuous robot body parameters using the value gradient of a frozen critic.

## Recommended Learning Order

1. First, study Build123d to understand parameters, primitives, boolean operations, and file export.
2. Then, study Text-to-CAD to understand how code agents participate in modifying CAD source files and performing geometric checks.
3. Next, study ForgeCAD to experience code-first CAD, official benchmark replication, and complex assembly presentation processes within the JavaScript/TypeScript ecosystem.
4. Finally, read Shape Your Body to understand how CAD/URDF/MJCF parameters further integrate into the learning robot design chain: “Body Parameters -> Simulation Control -> Value Gradient -> Design Diagnosis”.
