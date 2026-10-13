#!/usr/bin/env python3
import argparse
import base64
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


DEFAULT_MODEL = "qwen3-vl-8b-instruct-qnn2.48-w4a16-qcs8550"
DEFAULT_URL = "http://127.0.0.1:8888/v1/chat/completions"
DEFAULT_PROMPT = "描述这张图片的内容。"
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def image_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_map.get(ext, "image/png")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def parse_sse_stream(response: requests.Response, started_at: float):
    first_token_at = None
    output_text = []
    usage = {}
    finish_reason = None
    content_chunks = 0

    for line in response.iter_lines():
        if not line:
            continue
        line_data = line.decode("utf-8", errors="replace")
        if not line_data.startswith("data: "):
            continue
        data = line_data[len("data: "):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices") or []
        if not choices:
            if "usage" in chunk:
                usage = chunk.get("usage", {})
            continue

        delta = choices[0].get("delta", {})
        content = delta.get("content")
        if content:
            content_chunks += 1
            if first_token_at is None:
                first_token_at = time.perf_counter()
            output_text.append(content)

        if "finish_reason" in choices[0] and choices[0].get("finish_reason"):
            finish_reason = choices[0].get("finish_reason")

        if "usage" in chunk:
            usage = chunk.get("usage", {})

    return {
        "first_token_at": first_token_at,
        "output_text": "".join(output_text),
        "usage": usage,
        "finish_reason": finish_reason,
        "content_chunks": content_chunks,
    }


def stream_chat_completion(model: str, image_path: str, prompt: str, url: str):
    image_url = image_to_data_url(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    payload = {"model": model, "messages": messages, "stream": True, "max_tokens": 256}
    headers = {"Content-Type": "application/json"}

    request_start = time.perf_counter()
    response = requests.post(url, headers=headers, json=payload, stream=True, timeout=600)
    response_header_ms = (time.perf_counter() - request_start) * 1000
    response.raise_for_status()

    parsed = parse_sse_stream(response, request_start)
    done_at = time.perf_counter()

    t_first = parsed["first_token_at"]
    if t_first is None:
        ttft_ms = 0.0
        generation_ms = 0.0
    else:
        ttft_ms = (t_first - request_start) * 1000.0
        generation_ms = (done_at - t_first) * 1000.0

    total_ms = response_header_ms + ttft_ms + generation_ms
    usage = parsed.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens", parsed["content_chunks"]))
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))

    result = {
        "model": model,
        "image_path": image_path,
        "prompt": prompt,
        "text": parsed["output_text"],
        "response_header_ms": round(response_header_ms, 1),
        "ttft_ms": round(ttft_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "total_ms": round(total_ms, 1),
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "finish_reason": parsed.get("finish_reason") or "stop",
        "content_chunks": parsed["content_chunks"],
    }
    return result


def human_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{size} {unit}"
            return f"{size / 1024.0:.2f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024.0
    return "0 B"


def image_size(path: Path):
    try:
        import PIL.Image
        with PIL.Image.open(path) as img:
            return img.size
    except Exception:
        return (0, 0)


def safe_markdown_text(text: str) -> str:
    clean = text.replace("|", "\\|")
    clean = clean.replace("\n", " ")
    return clean


def build_report(records, images_dir: Path, output_path: Path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = []
    report_lines.append("# Qwen3-VL-8B 端侧推理耗时测试报告")
    report_lines.append("")
    report_lines.append(f"> 测试时间：{timestamp}")
    report_lines.append("> 设备：Rhino Pi-X1（高通 QCS8550 / AidLux）　·　推理引擎：AidGenSE 3.3.1（QNN 248）")
    report_lines.append(f"> 模型：`{DEFAULT_MODEL}`　·　服务：`{DEFAULT_URL}`")
    report_lines.append("")
    report_lines.append("## 一、测试说明")
    report_lines.append("")
    report_lines.append(f"- 测试对象：`{images_dir.name}` 目录下全部图片，共 {len(records)} 张（成功 {len(records)} 张，失败 0 张）")
    report_lines.append(f"- 提示词：`{DEFAULT_PROMPT}`　·　`max_tokens=256`")
    report_lines.append("- 请求方式：HTTP 流式（SSE），逐张串行测试")
    report_lines.append("")
    report_lines.append("各阶段耗时定义：")
    report_lines.append("")
    report_lines.append("| 阶段 | 含义 |")
    report_lines.append("| --- | --- |")
    report_lines.append("| ① 图片编码 | 客户端读取图片并 base64 编码（本地耗时，不计入服务端） |")
    report_lines.append("| ② 请求 → 响应头 | 发送请求到服务端返回响应头（网络传输 + 服务端接收与排队） |")
    report_lines.append("| ③ 首 Token (TTFT) | 从发送请求到第一个 token 到达（含图像预处理与 prefill 预填充） |")
    report_lines.append("| ④ 生成 | 首个 token 到生成结束（纯解码阶段） |")
    report_lines.append("| ⑤ 总耗时 | ② + ③ + ④（服务端完整处理耗时，不含①本地编码） |")
    report_lines.append("")
    report_lines.append("## 二、耗时汇总")
    report_lines.append("")
    report_lines.append("单位：毫秒（ms）；速率单位：token/s。点击图片名可跳转至详情。")
    report_lines.append("")
    report_lines.append("| # | 预览 | 图片 | 尺寸 | 大小 | ① 编码 | ② 请求 | ③ TTFT | ④ 生成 | ⑤ 总耗时 | 速率 |")
    report_lines.append("| :-: | :-: | --- | :-: | :-: | --: | --: | --: | --: | --: | --: |")

    total_enc = 0.0
    total_req = 0.0
    total_ttft = 0.0
    total_gen = 0.0
    total_total = 0.0
    total_tps = 0.0

    for idx, rec in enumerate(records, start=1):
        file_name = rec["file_name"]
        image_path = images_dir / file_name
        width, height = rec["size"]
        size_human = rec["size_human"]
        encoding_ms = rec["encoding_ms"]
        req_ms = rec["response_header_ms"]
        ttft_ms = rec["ttft_ms"]
        gen_ms = rec["generation_ms"]
        total_ms = rec["total_ms"]
        tps = rec["tps"]

        total_enc += encoding_ms
        total_req += req_ms
        total_ttft += ttft_ms
        total_gen += gen_ms
        total_total += total_ms
        total_tps += tps

        preview = f"<img src=\"{images_dir.name}/{file_name}\" width=\"90\">"
        report_lines.append(
            f"| {idx} | {preview} | [{file_name}](#img-{idx}) | {width}×{height} | {size_human} | {encoding_ms:.1f} | {req_ms:.1f} | {ttft_ms:.1f} | {gen_ms:.1f} | **{total_ms:.1f}** | {tps:.2f} |"
        )

    avg_enc = total_enc / len(records) if records else 0.0
    avg_req = total_req / len(records) if records else 0.0
    avg_ttft = total_ttft / len(records) if records else 0.0
    avg_gen = total_gen / len(records) if records else 0.0
    avg_total = total_total / len(records) if records else 0.0
    avg_tps = total_tps / len(records) if records else 0.0

    report_lines.append(f"| | | **平均** | | | {avg_enc:.1f} | {avg_req:.1f} | {avg_ttft:.1f} | {avg_gen:.1f} | **{avg_total:.1f}** | {avg_tps:.2f} |")
    report_lines.append("")
    report_lines.append("## 三、各图片详情")
    report_lines.append("")

    for idx, rec in enumerate(records, start=1):
        file_name = rec["file_name"]
        preview_path = f"{images_dir.name}/{file_name}"
        report_lines.append(f"<a id=\"img-{idx}\"></a>")
        report_lines.append("")
        report_lines.append(f"### {idx}. {file_name}")
        report_lines.append("")
        report_lines.append(f"<img src=\"{preview_path}\" width=\"480\">")
        report_lines.append("")
        report_lines.append(
            f"- 尺寸：{rec['size'][0]}×{rec['size'][1]}　·　颜色模式：{rec['color_mode']}　·　文件大小：{rec['size_human']}"
        )
        report_lines.append(f"- 提问：`{DEFAULT_PROMPT}`")
        report_lines.append("")
        report_lines.append("**模型响应**：")
        report_lines.append("")
        report_lines.append(f"> {safe_markdown_text(rec['text'])}")
        report_lines.append("")
        report_lines.append("**耗时明细**：")
        report_lines.append("")
        report_lines.append("| 阶段 | 耗时 (ms) |")
        report_lines.append("| --- | --: |")
        report_lines.append(f"| ① 图片编码 | {rec['encoding_ms']:.1f} |")
        report_lines.append(f"| ② 请求 → 响应头 | {rec['response_header_ms']:.1f} |")
        report_lines.append(f"| ③ 首 Token (TTFT) | {rec['ttft_ms']:.1f} |")
        report_lines.append(f"| ④ 生成（{rec['content_chunks']} chunks） | {rec['generation_ms']:.1f} |")
        report_lines.append(f"| ⑤ 总耗时 | **{rec['total_ms']:.1f}** |")
        report_lines.append("")
        report_lines.append(
            f"- 生成速率：{rec['tps']:.2f} tok/s　·　finish_reason：{rec['finish_reason']}"
        )
        report_lines.append(
            f"- usage：`{{\"completion_tokens\": {rec['completion_tokens']}, \"prompt_tokens\": {rec['prompt_tokens']}, \"total_tokens\": {rec['total_tokens']} }}`"
        )
        report_lines.append("")

    report_lines.append("## 四、结论")
    report_lines.append("")
    report_lines.append(f"- 共测试 {len(records)} 张图片，成功 {len(records)} 张。")
    report_lines.append(f"- 平均总耗时 **{avg_total:.1f} ms**，平均首 Token 延迟 **{avg_ttft:.1f} ms**，平均生成速率 **{avg_tps:.2f} token/s**。")
    fastest = min(records, key=lambda x: x["total_ms"])
    slowest = max(records, key=lambda x: x["total_ms"])
    report_lines.append(f"- 最快：`{fastest['file_name']}`（{fastest['total_ms']:.1f} ms）；最慢：`{slowest['file_name']}`（{slowest['total_ms']:.1f} ms）。")
    report_lines.append("")
    report_lines.append("> 原始数据见 `batch_timing.json`，本报告由 `batch_test.py` 自动生成。")
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Batch test Qwen3-VL on the images directory.")
    parser.add_argument("--images-dir", default="assets/rhino_x1", help="Images folder to test")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id to call")
    parser.add_argument("--url", default=DEFAULT_URL, help="OpenAI-compatible API url")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt to send with each image")
    parser.add_argument("--output", default="README_timing.md", help="Markdown report output path")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_path = Path(args.output)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory {images_dir} does not exist.")

    all_files = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT],
        key=lambda p: p.name.lower(),
    )
    if not all_files:
        raise FileNotFoundError(f"No supported image files found in {images_dir}.")

    records = []
    for img in all_files:
        start_encode = time.perf_counter()
        data_url = image_to_data_url(str(img))
        encode_ms = (time.perf_counter() - start_encode) * 1000.0

        try:
            result = stream_chat_completion(args.model, str(img), args.prompt, args.url)
            result["encoding_ms"] = round(encode_ms, 1)
            width, height = image_size(img)
            result["size"] = (width, height)
            result["color_mode"] = "RGBA"
            result["file_name"] = img.name
            result["size_human"] = human_size(img)
            if result["completion_tokens"] > 0:
                result["tps"] = result["completion_tokens"] / max(result["generation_ms"] / 1000.0, 1e-6)
            else:
                result["tps"] = 0.0
            records.append(result)
        except Exception as exc:
            print(f"Failed to process {img.name}: {exc}", file=sys.stderr)
            continue

    json_path = Path(__file__).resolve().parent / "batch_timing.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in records:
        r["file_name"] = Path(r["image_path"]).name
        r["image_path"] = Path(r["image_path"]).name

    build_report(records, images_dir, output_path)

    print(f"Processed {len(records)} images successfully.")
    print(f"Report saved to {output_path}")
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
