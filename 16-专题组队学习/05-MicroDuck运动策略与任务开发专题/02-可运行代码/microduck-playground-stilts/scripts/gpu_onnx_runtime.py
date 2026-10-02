#!/usr/bin/env python3
"""Small Torch/CUDA executor for the exported MicroDuck ONNX actor graphs.

The teaching Ubuntu image has CUDA PyTorch but only the CPU ONNX Runtime
provider.  The exported actors in this topic are intentionally small, so this
module evaluates their ONNX graph with Torch tensors on the available CUDA
device.  It covers the operators emitted by the walking, basketball and
FastSAC export paths and keeps the policy contract visible in the video code.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch


def _attribute(node: onnx.NodeProto, name: str, default=None):
    for item in node.attribute:
        if item.name != name:
            continue
        if item.type == onnx.AttributeProto.INT:
            return int(item.i)
        if item.type == onnx.AttributeProto.INTS:
            return tuple(int(value) for value in item.ints)
        if item.type == onnx.AttributeProto.FLOAT:
            return float(item.f)
        if item.type == onnx.AttributeProto.FLOATS:
            return tuple(float(value) for value in item.floats)
        if item.type == onnx.AttributeProto.TENSOR:
            return torch.from_numpy(numpy_helper.to_array(item.t).copy()).float()
        if item.type == onnx.AttributeProto.STRING:
            return item.s.decode("utf-8")
    return default


class TorchOnnxPolicy:
    """Run one exported actor graph with Torch tensors on CUDA."""

    def __init__(self, model: Path, device: str | None = None):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU 推理需要可用的 CUDA PyTorch 环境。")
        self.device = torch.device(device or "cuda")
        if self.device.type != "cuda":
            raise ValueError("GPU 推理脚本只接受 CUDA 设备。")
        self.model_path = Path(model).resolve()
        graph = onnx.load(str(self.model_path)).graph
        self.nodes = list(graph.node)
        self.initializers = {
            item.name: torch.from_numpy(numpy_helper.to_array(item).copy()).to(
                device=self.device, dtype=torch.float32
            )
            for item in graph.initializer
        }
        self.input_names = [item.name for item in graph.input]
        self.output_names = [item.name for item in graph.output]
        self.latencies_ms: list[float] = []
        self.hidden: torch.Tensor | None = None
        self.cell: torch.Tensor | None = None

    def reset(self) -> None:
        self.hidden = None
        self.cell = None

    def _value(self, name: str, values: dict[str, torch.Tensor]) -> torch.Tensor:
        if name in values:
            return values[name]
        if name in self.initializers:
            return self.initializers[name]
        raise KeyError(f"ONNX value is unavailable: {name}")

    def _axes(self, node: onnx.NodeProto, values: dict[str, torch.Tensor], index: int):
        attr = _attribute(node, "axes", None)
        if attr is not None:
            return tuple(int(value) for value in attr)
        if len(node.input) > index and node.input[index]:
            return tuple(int(value) for value in self._value(node.input[index], values).flatten().tolist())
        return None

    def _lstm(self, node: onnx.NodeProto, values: dict[str, torch.Tensor]):
        x = self._value(node.input[0], values)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        w = self._value(node.input[1], values)[0]
        r = self._value(node.input[2], values)[0]
        bias = self._value(node.input[3], values)[0]
        hidden_size = int(_attribute(node, "hidden_size", r.shape[1] // 4))
        batch = x.shape[1]
        h = self.hidden if self.hidden is not None else torch.zeros(
            (batch, hidden_size), device=self.device, dtype=x.dtype
        )
        c = self.cell if self.cell is not None else torch.zeros_like(h)
        if len(node.input) > 5 and node.input[5]:
            h = self._value(node.input[5], values)[0]
        if len(node.input) > 6 and node.input[6]:
            c = self._value(node.input[6], values)[0]
        wb, rb = bias[: 4 * hidden_size], bias[4 * hidden_size :]
        outputs = []
        for xt in x:
            gates = xt @ w.transpose(0, 1) + h @ r.transpose(0, 1) + wb + rb
            gate_i, gate_o, gate_f, gate_g = gates.chunk(4, dim=-1)
            gate_i = torch.sigmoid(gate_i)
            gate_o = torch.sigmoid(gate_o)
            gate_f = torch.sigmoid(gate_f)
            gate_g = torch.tanh(gate_g)
            c = gate_f * c + gate_i * gate_g
            h = gate_o * torch.tanh(c)
            outputs.append(h)
        y = torch.stack(outputs, dim=0).unsqueeze(1)
        self.hidden = h.detach()
        self.cell = c.detach()
        return y, h.unsqueeze(0), c.unsqueeze(0)

    def _run(self, feeds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values = dict(feeds)
        for node in self.nodes:
            op = node.op_type
            args = [self._value(name, values) for name in node.input if name]
            if op == "Constant":
                result = [_attribute(node, "value")]
                if result[0] is None:
                    result = [torch.tensor(_attribute(node, "value_float", _attribute(node, "value_int", 0)), device=self.device)]
                elif isinstance(result[0], torch.Tensor):
                    result[0] = result[0].to(device=self.device)
            elif op == "Identity":
                result = [args[0]]
            elif op in {"Add", "Sub", "Mul", "Div", "Pow", "MatMul"}:
                result = [{"Add": torch.add, "Sub": torch.sub, "Mul": torch.mul, "Div": torch.div, "Pow": torch.pow, "MatMul": torch.matmul}[op](args[0], args[1])]
            elif op == "Gemm":
                alpha = float(_attribute(node, "alpha", 1.0))
                beta = float(_attribute(node, "beta", 1.0))
                a = args[0].transpose(-1, -2) if int(_attribute(node, "transA", 0)) else args[0]
                b = args[1].transpose(-1, -2) if int(_attribute(node, "transB", 0)) else args[1]
                result = [alpha * (a @ b) + beta * args[2]]
            elif op == "Elu":
                result = [torch.nn.functional.elu(args[0], alpha=float(_attribute(node, "alpha", 1.0)))]
            elif op == "Relu":
                result = [torch.relu(args[0])]
            elif op == "Tanh":
                result = [torch.tanh(args[0])]
            elif op == "Sigmoid":
                result = [torch.sigmoid(args[0])]
            elif op == "Sqrt":
                result = [torch.sqrt(args[0])]
            elif op == "ReduceMean":
                axes = self._axes(node, values, 1)
                result = [torch.mean(args[0], dim=axes, keepdim=bool(_attribute(node, "keepdims", 1)))] if axes else [torch.mean(args[0])]
            elif op == "ReduceSum":
                axes = self._axes(node, values, 1)
                result = [torch.sum(args[0], dim=axes, keepdim=bool(_attribute(node, "keepdims", 1)))] if axes else [torch.sum(args[0])]
            elif op == "Unsqueeze":
                out = args[0]
                for axis in sorted(self._axes(node, values, 1) or (),):
                    out = torch.unsqueeze(out, axis)
                result = [out]
            elif op == "Squeeze":
                out = args[0]
                axes = self._axes(node, values, 1)
                if axes is None:
                    out = torch.squeeze(out)
                else:
                    for axis in sorted(axes, reverse=True):
                        out = torch.squeeze(out, axis)
                result = [out]
            elif op == "LSTM":
                result = list(self._lstm(node, values))
            elif op == "Clip":
                lower = float(args[1].item()) if len(args) > 1 else float(_attribute(node, "min", -float("inf")))
                upper = float(args[2].item()) if len(args) > 2 else float(_attribute(node, "max", float("inf")))
                result = [torch.clamp(args[0], min=lower, max=upper)]
            elif op == "Reshape":
                shape = tuple(int(value) for value in args[1].flatten().tolist())
                result = [args[0].reshape(shape)]
            elif op == "Flatten":
                axis = int(_attribute(node, "axis", 1))
                result = [args[0].flatten(start_dim=axis)]
            elif op == "Transpose":
                result = [args[0].permute(tuple(_attribute(node, "perm", tuple(range(args[0].ndim)))))]
            elif op == "Concat":
                result = [torch.cat(args, dim=int(_attribute(node, "axis", 0)))]
            elif op == "Cast":
                result = [args[0].float()]
            elif op == "Shape":
                result = [torch.tensor(list(args[0].shape), device=self.device, dtype=torch.int64)]
            elif op == "Gather":
                axis = int(_attribute(node, "axis", 0))
                result = [torch.index_select(args[0], axis, args[1].to(torch.long).flatten())]
            else:
                raise NotImplementedError(f"TorchOnnxPolicy does not support ONNX op {op}")
            for name, value in zip(node.output, result):
                values[name] = value
        return values

    def infer(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError("non-finite actor observation")
        feeds = {"obs": torch.from_numpy(value).reshape(1, -1).to(self.device)}
        if "h_in" in self.input_names:
            feeds["h_in"] = torch.zeros((1, 1, 256), device=self.device)
            feeds["c_in"] = torch.zeros((1, 1, 256), device=self.device)
            if self.hidden is not None:
                feeds["h_in"] = self.hidden.unsqueeze(0)
                feeds["c_in"] = self.cell.unsqueeze(0)
        started = time.perf_counter()
        values = self._run(feeds)
        torch.cuda.synchronize(self.device)
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        action = values[self.output_names[0]].detach().float().cpu().numpy().reshape(-1)
        if not np.isfinite(action).all():
            raise RuntimeError("GPU ONNX returned a non-finite action")
        return action

    def close(self) -> None:
        self.reset()
