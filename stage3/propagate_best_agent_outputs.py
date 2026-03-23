#!/usr/bin/env python3
"""Propagate masks using the best available agent stage per group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage2.generate_sam3_masks_from_agent import process_group
from stage2.generate_sam3_masks_from_bbox import DEFAULT_CHECKPOINT, build_video_tracker, import_sam3_builders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Choose the best available agent stage per group (initial, refine1, fallback) "
            "according to scan reports and propagate masks with the official SAM3 tracker."
        )
    )
    parser.add_argument("--base-batch-root", default="outputs/batch_ark_video_qa")
    parser.add_argument("--initial-agent-root", required=True)
    parser.add_argument("--initial-scan-json", required=True)
    parser.add_argument("--refine1-agent-root", required=True)
    parser.add_argument("--refine1-scan-json", required=True)
    parser.add_argument("--fallback-agent-root", required=True)
    parser.add_argument("--fallback-scan-json", required=True)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--mask-root", default="mask_output")
    parser.add_argument("--exp-name", default="exp_agent_final")
    parser.add_argument("--sam3-official-root", default="sam3_official")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-thresh", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_bad_group_set(path: Path) -> set[tuple[str, str]]:
    data = json.loads(path.read_text())
    return {(item["video_id"], str(item["json_id"])) for item in data.get("group_badcases", [])}


def load_terminal_empty_group_set(path: Path) -> set[tuple[str, str]]:
    data = json.loads(path.read_text())
    return {(item["video_id"], str(item["json_id"])) for item in data.get("terminal_empty_groups", [])}


def list_base_groups(batch_root: Path) -> list[tuple[str, str]]:
    groups = []
    for path in sorted(batch_root.glob("*/*.json")):
        if path.parent.name == "_errors":
            continue
        groups.append((path.parent.name, path.stem))
    return groups


def pred_paths_for_group(agent_root: Path, video_id: str, json_id: str) -> list[Path]:
    group_dir = agent_root / video_id / json_id
    return sorted(group_dir.glob("*_obj_*/pred.json"))


def main() -> int:
    args = parse_args()
    base_groups = list_base_groups(Path(args.base_batch_root))
    initial_bad = load_bad_group_set(Path(args.initial_scan_json))
    initial_terminal_empty = load_terminal_empty_group_set(Path(args.initial_scan_json))
    refine1_bad = load_bad_group_set(Path(args.refine1_scan_json))
    fallback_bad = load_bad_group_set(Path(args.fallback_scan_json))

    selected_groups: list[dict[str, Any]] = []
    last_list: list[dict[str, str]] = []
    terminal_empty_groups: list[dict[str, str]] = []
    for video_id, json_id in base_groups:
        key = (video_id, json_id)
        if key in initial_terminal_empty:
            terminal_empty_groups.append({"video_id": video_id, "json_id": json_id})
            continue
        if key not in initial_bad:
            root = Path(args.initial_agent_root)
            stage = "initial"
        elif key not in refine1_bad:
            root = Path(args.refine1_agent_root)
            stage = "refine1"
        elif key not in fallback_bad:
            root = Path(args.fallback_agent_root)
            stage = "fallback"
        else:
            last_list.append({"video_id": video_id, "json_id": json_id})
            continue

        pred_paths = pred_paths_for_group(root, video_id, json_id)
        if not pred_paths:
            last_list.append({"video_id": video_id, "json_id": json_id})
            continue
        selected_groups.append(
            {
                "video_id": video_id,
                "json_id": json_id,
                "pred_paths": pred_paths,
                "selected_stage": stage,
            }
        )

    build_sam3_image_model, build_sam3_video_model = import_sam3_builders(Path(args.sam3_official_root))
    tracker = build_video_tracker(build_sam3_video_model, args.checkpoint, args.device)
    output_root = Path(args.mask_root) / args.exp_name

    success_count = 0
    error_count = 0
    selection_summary = []
    for group in selected_groups:
        prefix = f"{group['video_id']}/{group['json_id']}"
        try:
            summary = process_group(
                group,
                dataset_root=Path(args.dataset_root),
                output_root=output_root,
                tracker=tracker,
                score_thresh=args.score_thresh,
                overwrite=args.overwrite,
            )
            success_count += 1
            selection_summary.append(
                {
                    "video_id": group["video_id"],
                    "json_id": group["json_id"],
                    "selected_stage": group["selected_stage"],
                    "output_root": summary["output_root"],
                }
            )
            print(f"saved {prefix} ({group['selected_stage']}) -> {summary['output_root']}")
        except Exception as exc:
            error_count += 1
            print(f"error {prefix}: {type(exc).__name__}: {exc}")

    final_summary = {
        "selected_group_count": len(selected_groups),
        "last_list_count": len(last_list),
        "terminal_empty_group_count": len(terminal_empty_groups),
        "success_count": success_count,
        "error_count": error_count,
        "output_root": str(output_root.resolve()),
        "selection_summary": selection_summary,
        "last_list": last_list,
        "terminal_empty_groups": terminal_empty_groups,
    }
    summary_path = output_root / "_selection_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print(f"saved_selection_summary: {summary_path}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
