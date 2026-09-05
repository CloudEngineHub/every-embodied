# Microduck Pages overlay

GitHub Actions applies this small source overlay to the pinned upstream
Microduck Simulator revision during the Pages build. Models, robot meshes,
MuJoCo WASM, `node_modules`, and generated `dist/` files are deliberately not
stored in the Every Embodied Git history.

The public Pages demo defaults to browser-side ONNX inference. The optional
RDK provider activates only when a developer explicitly supplies an `rdk`
query parameter and runs the local bridge described in the Chinese tutorial.

Upstream project:
<https://huggingface.co/spaces/pollen-robotics/microduck-simulator>

Pinned upstream commit:
`e81974b932c7ca1819843b7bb3dcd42e2993e98e`
