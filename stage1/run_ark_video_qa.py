#!/usr/bin/env python3
"""Run a single multi-image QA request against Ark using dataset video frames."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-plus"
AUTO_DOWNSAMPLE_THRESHOLD = 70
AUTO_DOWNSAMPLE_STRIDE = 2
JSON_OUTPUT_INSTRUCTION = """## Output Format

你必须只输出一个 JSON 数组，不允许输出任何额外解释、标题、注释或 Markdown 代码块。

数组中的每个元素都必须是如下结构的 JSON 对象：
{
  "object_name": "用于唯一标识该实体的名称",
  "best_frame": 3,
  "discriminative_description": "基于该秒全局画面的对主语的唯一判别性描述"
}

额外要求：
1. `best_frame` 必须是整数，表示你选择的秒数。
2. 如果存在同时满足目标事件但不同的实体，`object_name` 必须显式区分，且描述中禁止提及其它主体。
3. 如果视频中不存在与目标事件直接相关的实体，输出空数组 `[]`。
4. 输出必须是合法 JSON，可被 `json.loads` 直接解析。"""
EVENT_PROMPT_TEMPLATE = """# Role
你是一个顶级的视频内容分析与视觉定位专家。你的任务是精准拆解视频中的特定事件，提取所有相关物体，并在时间和空间上提供唯一的、具有判别性的描述。

# Inputs
- 目标事件：{target_event}

## Task Instructions

请仔细观看视频，并严格围绕“目标事件”，执行以下步骤：

1. 识别所有独立的物体实例（Instances）：仔细分析目标事件，找出该短语真正想要指代的**核心主体（中心词）**。
   极度重要：必须进行实例级别的区分！你只需提取核心主体，绝对禁止提取参照物和修饰语！如果视频中出现了同时符合目标事件但非同一核心主体的物体，你必须将它们视为两个独立的物体分别提取。如果不符合修饰语，则视为无关物体，不予提取。
   反之，如果同一个物理实体在不同时间段多次出现，请将其视为一个物体处理。
2. 选取最佳时间帧：对于找出的每一个独立的物理实体，评估其在视频中出现的所有时间。请仅挑选一秒该实体特征最明显、最无遮挡、最易于辨认的时间（必须为整数秒）。
3. 撰写判别性描述：在选定的这一秒画面全景中，为该实体生成一句唯一且具有判别性（Discriminative）的描述。描述必须包含明确的视觉属性（颜色、形状、材质、人物穿着）和空间相对位置（左上角、在XX旁边），确保仅凭这句话就能在该秒画面中唯一锁定该物体，没有任何歧义。

## Strict Constraints

1. 唯一性与实例区分：每个独立的物理实体仅输出一条记录，绝对不能对同一实体重复提取。如果是不同的实体，必须分别输出。
2. 命名规范：对于同类但不同的实体，请在 object_name 中加以明确区分。
3. 完整性：必须穷尽所有与该事件直接相关的独立物体，不要遗漏。
4. 时间格式：必须是唯一的整数秒（例如：3，而不是 \"3s\" 或 \"2-4\"）。
5. 描述要求：必须基于“选定秒”的全局画面进行判别性描述。禁止使用泛泛的词汇。"""


@dataclass
class RequestBundle:
    video_id: str
    prompt_text: str
    payload: dict[str, Any]
    selected_frames: list[tuple[int, Path]]
    selection_strategy: str
    total_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one Ark chat-completions request from dataset/JPEGImages/<video_id>. "
            "If a video has more than 70 frames, the script keeps every other frame "
            "while still labeling them [0.0 second], [1.0 second], ..."
        )
    )
    parser.add_argument("--video-id", required=True, help="Video id under dataset/JPEGImages")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--question",
        help="Direct prompt text to ask about the video",
    )
    prompt_group.add_argument(
        "--target-event",
        help="Fill the built-in event-analysis prompt template with this target event",
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Dataset root directory containing JPEGImages/",
    )
    parser.add_argument(
        "--response-format",
        choices=("json", "text"),
        default="json",
        help="json: append a strict JSON output contract; text: leave the prompt unchanged",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ark model name")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ark API base URL")
    parser.add_argument(
        "--save-payload",
        help="Optional path to save the generated payload JSON before sending",
    )
    parser.add_argument(
        "--save-response",
        help="Optional path to save the normalized response JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build and preview the messages without calling the API",
    )
    return parser.parse_args()


def build_prompt_text(
    *,
    question: str | None = None,
    target_event: str | None = None,
    response_format: str = "json",
) -> str:
    if question:
        prompt_text = question
    elif target_event:
        prompt_text = EVENT_PROMPT_TEMPLATE.format(target_event=target_event)
    else:
        raise ValueError("Either question or target_event is required")

    if response_format == "json":
        prompt_text = f"{prompt_text}\n\n{JSON_OUTPUT_INSTRUCTION}"
    return prompt_text


def list_frames(video_dir: Path) -> list[Path]:
    frames = sorted(video_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not frames:
        raise FileNotFoundError(f"No .jpg frames found in {video_dir}")
    return frames


def select_frames(frames: list[Path]) -> tuple[list[tuple[int, Path]], str]:
    if len(frames) > AUTO_DOWNSAMPLE_THRESHOLD:
        selected = [(idx, frames[idx]) for idx in range(0, len(frames), AUTO_DOWNSAMPLE_STRIDE)]
        strategy = f"every_{AUTO_DOWNSAMPLE_STRIDE}_frames (total>{AUTO_DOWNSAMPLE_THRESHOLD})"
        return selected, strategy
    return list(enumerate(frames)), "all_frames"


def format_timestamp(sample_order: int) -> str:
    return f"[{float(sample_order):.1f} second]"


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_content(prompt_text: str, selected_frames: list[tuple[int, Path]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for sample_order, (_, frame_path) in enumerate(selected_frames):
        content.append({"type": "text", "text": format_timestamp(sample_order)})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(frame_path)},
                # "detail": "low",
            }
        )
    return content


def prepare_request(
    *,
    video_id: str,
    dataset_root: str | Path,
    prompt_text: str,
    model: str = DEFAULT_MODEL,
) -> RequestBundle:
    dataset_root = Path(dataset_root)
    video_dir = dataset_root / "JPEGImages" / video_id
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video id {video_id!r} not found under {video_dir}")

    frames = list_frames(video_dir)
    selected_frames, strategy = select_frames(frames)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": build_content(prompt_text, selected_frames),
            }
        ],
    }
    return RequestBundle(
        video_id=video_id,
        prompt_text=prompt_text,
        payload=payload,
        selected_frames=selected_frames,
        selection_strategy=strategy,
        total_frames=len(frames),
    )


def save_json(data: Any, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    preview = {
        "model": payload["model"],
        "messages": [
            {
                "role": "user",
                "content": [],
            }
        ],
    }
    for item in payload["messages"][0]["content"]:
        if item["type"] == "image_url":
            preview["messages"][0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"<data_url length={len(item['image_url']['url'])}>"},
                }
            )
        else:
            preview["messages"][0]["content"].append(item)
    return preview


def preview_selected_frames(selected_frames: list[tuple[int, Path]]) -> None:
    print("selected_frames:")
    preview_limit = 8
    if len(selected_frames) <= preview_limit * 2:
        iterable = list(enumerate(selected_frames))
    else:
        iterable = list(enumerate(selected_frames[:preview_limit]))
        iterable.append((-1, (-1, Path("..."))))
        offset = len(selected_frames) - preview_limit
        iterable.extend((offset + i, item) for i, item in enumerate(selected_frames[-preview_limit:]))

    for sample_order, item in iterable:
        if sample_order == -1:
            print("  - ...")
            continue
        frame_index, frame_path = item
        print(f"  - {format_timestamp(sample_order)} original_frame={frame_index} path={frame_path}")


def preview_request(
    *,
    video_id: str,
    prompt_label: str,
    prompt_value: str,
    bundle: RequestBundle,
) -> None:
    print(f"video_id: {video_id}")
    print(f"{prompt_label}: {prompt_value}")
    print("prompt_text:")
    print(bundle.prompt_text)
    print(f"total_frames: {bundle.total_frames}")
    print(f"selection_strategy: {bundle.selection_strategy}")
    print(f"selected_images: {len(bundle.selected_frames)}")
    preview_selected_frames(bundle.selected_frames)
    print("\npayload_preview:")
    print(json.dumps(sanitize_payload(bundle.payload), ensure_ascii=False, indent=2))


def extract_answer(response: Any) -> str:
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                texts.append(text)
            elif isinstance(item, dict) and item.get("text"):
                texts.append(item["text"])
        return "\n".join(texts)
    return str(content)


def extract_json_block(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_markers = ("```json", "```JSON", "```")
    for marker in fence_markers:
        if marker in text:
            start = text.find(marker)
            body = text[start + len(marker) :]
            end = body.find("```")
            if end != -1:
                candidate = body[:end].strip()
                if candidate:
                    return json.loads(candidate)

    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        return json.loads(text[array_start : array_end + 1])

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        return json.loads(text[object_start : object_end + 1])

    raise ValueError("No JSON object/array found in response")


def _coerce_best_frame(record: dict[str, Any]) -> int:
    if "best_frame" in record:
        value = record["best_frame"]
    elif "best_frame_time" in record:
        value = record["best_frame_time"]
    else:
        raise ValueError("Missing best_frame field")
    if isinstance(value, bool):
        raise ValueError("best_frame cannot be bool")
    return int(value)


def normalize_result_records(answer_text: str, selected_frames: list[tuple[int, Path]]) -> list[dict[str, Any]]:
    parsed = extract_json_block(answer_text)
    if isinstance(parsed, dict):
        if "results" in parsed and isinstance(parsed["results"], list):
            parsed = parsed["results"]
        else:
            raise ValueError("Expected a JSON array or an object with a results array")
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array")

    normalized: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Each JSON item must be an object")
        record = dict(item)
        best_frame = _coerce_best_frame(record)
        if best_frame < 0 or best_frame >= len(selected_frames):
            raise ValueError(
                f"best_frame {best_frame} out of range for {len(selected_frames)} selected frames"
            )
        actual_frame_index, actual_frame_path = selected_frames[best_frame]
        record["best_frame"] = best_frame
        record["actual_frame_filename"] = actual_frame_path.name
        record["actual_frame_index"] = actual_frame_index
        normalized.append(record)
    return normalized


def build_result_payload(
    *,
    video_id: str,
    prompt_label: str,
    prompt_value: str,
    bundle: RequestBundle,
    results: list[dict[str, Any]] | None = None,
    raw_response_text: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "video_id": video_id,
        prompt_label: prompt_value,
        "selection_strategy": bundle.selection_strategy,
        "total_frames": bundle.total_frames,
        "selected_images": len(bundle.selected_frames),
    }
    if results is not None:
        payload["results"] = results
    if raw_response_text is not None:
        payload["raw_response_text"] = raw_response_text
    return payload


def format_response_text(
    answer_text: str,
    *,
    response_format: str,
    selected_frames: list[tuple[int, Path]],
) -> str:
    if response_format != "json":
        return answer_text
    parsed = normalize_result_records(answer_text, selected_frames)
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def call_api(payload: dict[str, Any], base_url: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for non-dry-run mode") from exc

    api_key = (
        os.environ.get("ARK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
    )
    if not api_key:
        raise RuntimeError("Set ARK_API_KEY, OPENAI_API_KEY, or DASHSCOPE_API_KEY before calling the API")

    client = OpenAI(base_url=base_url, api_key=api_key)
    return client.chat.completions.create(**payload)


def main() -> int:
    args = parse_args()
    prompt_label = "question" if args.question else "target_event"
    prompt_value = args.question if args.question else args.target_event
    prompt_text = build_prompt_text(
        question=args.question,
        target_event=args.target_event,
        response_format=args.response_format,
    )
    bundle = prepare_request(
        video_id=args.video_id,
        dataset_root=args.dataset_root,
        prompt_text=prompt_text,
        model=args.model,
    )

    if args.save_payload:
        save_json(bundle.payload, args.save_payload)
        print(f"saved payload to {args.save_payload}")

    preview_request(
        video_id=args.video_id,
        prompt_label=prompt_label,
        prompt_value=prompt_value,
        bundle=bundle,
    )
    if args.dry_run:
        return 0

    response = call_api(bundle.payload, args.base_url)
    answer = extract_answer(response)
    print("\nassistant_response:")
    try:
        print(
            format_response_text(
                answer,
                response_format=args.response_format,
                selected_frames=bundle.selected_frames,
            )
        )
    except Exception as exc:
        print(answer)
        print(f"\n[warning] failed to normalize JSON response: {exc}")

    if args.save_response:
        if args.response_format == "json":
            results = normalize_result_records(answer, bundle.selected_frames)
            result_payload = build_result_payload(
                video_id=args.video_id,
                prompt_label=prompt_label,
                prompt_value=prompt_value,
                bundle=bundle,
                results=results,
                raw_response_text=answer,
            )
        else:
            result_payload = build_result_payload(
                video_id=args.video_id,
                prompt_label=prompt_label,
                prompt_value=prompt_value,
                bundle=bundle,
                raw_response_text=answer,
            )
        save_json(result_payload, args.save_response)
        print(f"saved response to {args.save_response}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
