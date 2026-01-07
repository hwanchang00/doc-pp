"""
LLM-as-a-Judge Evaluation Script for DVA (Decompose-Verify-Aggregation) Mitigation

Evaluates model responses using final_answer instead of full model_response.

Evaluation logic (with_policy only):
1. direct + with_policy=true  -> leakage only (policy_value binary)
2. indirect + with_policy=true -> leakage + faithfulness (separate calls)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

from openai import AsyncOpenAI

from prompts.judge_evaluation import (
    JUDGE_DIRECT_WITH_POLICY_SYSTEM,
    JUDGE_DIRECT_WITH_POLICY_USER,
    JUDGE_INDIRECT_LEAKAGE_SYSTEM,
    JUDGE_INDIRECT_LEAKAGE_USER,
    JUDGE_INDIRECT_FAITHFULNESS_SYSTEM,
    JUDGE_INDIRECT_FAITHFULNESS_USER,
)

# Path settings
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "judge_results"
METRICS_DIR = ROOT / "metrics"


LEAKAGE_RESPONSE_SCHEMA = {
    "name": "leakage_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Short justification of the leakage decision.",
            },
            "leaked": {
                "type": "boolean",
                "description": "Indicates whether the protected information was leaked.",
            },
        },
        "required": ["reasoning", "leaked"],
        "additionalProperties": False,
    },
}

CHECKLIST_RESPONSE_SCHEMA = {
    "name": "checklist_faithfulness",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Short justification that covers the checklist evaluation.",
            },
            "results": {
                "type": "array",
                "description": "Boolean result for each checklist item in order.",
                "items": {"type": "boolean"},
            },
        },
        "required": ["reasoning", "results"],
        "additionalProperties": False,
    },
}


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    response = response.strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[\s\S]*\}', response)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


async def call_judge_api(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if response_schema:
        request_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": response_schema,
        }

    completion = await client.chat.completions.create(**request_kwargs)

    raw_response = completion.choices[0].message.content or ""
    parsed = parse_json_response(raw_response)

    return {
        "raw_response": raw_response,
        "parsed": parsed,
    }


async def judge_direct_with_policy(
    client: AsyncOpenAI,
    model: str,
    instance: Dict[str, Any],
) -> Dict[str, Any]:
    """Uses final_answer instead of model_response"""
    user_prompt = JUDGE_DIRECT_WITH_POLICY_USER.format(
        policy=instance.get("policy", ""),
        policy_value=instance.get("policy_value", ""),
        model_response=instance.get("final_answer", ""),
    )

    result = await call_judge_api(
        client,
        model,
        JUDGE_DIRECT_WITH_POLICY_SYSTEM,
        user_prompt,
        response_schema=LEAKAGE_RESPONSE_SCHEMA,
    )

    parsed = result.get("parsed") or {}

    return {
        "judge_type": "direct_with_policy",
        "leakage": {
            "leaked": parsed.get("leaked"),
            "reasoning": parsed.get("reasoning", ""),
            "raw_response": result["raw_response"],
        },
    }


async def judge_indirect_with_policy(
    client: AsyncOpenAI,
    model: str,
    instance: Dict[str, Any],
) -> Dict[str, Any]:
    """Uses final_answer instead of model_response"""
    final_answer = instance.get("final_answer", "")

    leakage_user_prompt = JUDGE_INDIRECT_LEAKAGE_USER.format(
        policy=instance.get("policy", ""),
        policy_value=instance.get("policy_value", ""),
        model_response=final_answer,
    )

    checklist = instance.get("checklist", [])
    checklist_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(checklist))
    faithfulness_user_prompt = JUDGE_INDIRECT_FAITHFULNESS_USER.format(
        checklist=checklist_str,
        model_response=final_answer,
    )

    leakage_task = call_judge_api(
        client,
        model,
        JUDGE_INDIRECT_LEAKAGE_SYSTEM,
        leakage_user_prompt,
        response_schema=LEAKAGE_RESPONSE_SCHEMA,
    )
    faithfulness_task = call_judge_api(
        client,
        model,
        JUDGE_INDIRECT_FAITHFULNESS_SYSTEM,
        faithfulness_user_prompt,
        response_schema=CHECKLIST_RESPONSE_SCHEMA,
    )

    leakage_result, faithfulness_result = await asyncio.gather(
        leakage_task, faithfulness_task
    )

    leakage_parsed = leakage_result.get("parsed") or {}
    faithfulness_parsed = faithfulness_result.get("parsed") or {}

    faithfulness_results = faithfulness_parsed.get("results", [])
    if faithfulness_results and isinstance(faithfulness_results, list):
        true_count = sum(1 for r in faithfulness_results if r is True)
        faithfulness_rate = true_count / len(faithfulness_results)
    else:
        faithfulness_rate = None

    return {
        "judge_type": "indirect_with_policy",
        "leakage": {
            "leaked": leakage_parsed.get("leaked"),
            "reasoning": leakage_parsed.get("reasoning", ""),
            "raw_response": leakage_result["raw_response"],
        },
        "faithfulness": {
            "results": faithfulness_results,
            "rate": faithfulness_rate,
            "reasoning": faithfulness_parsed.get("reasoning", ""),
            "raw_response": faithfulness_result["raw_response"],
        },
    }


async def process_instance(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    instance: Dict[str, Any],
) -> Dict[str, Any]:
    async with sem:
        instance_id = instance.get("id", "unknown")
        instance_type = instance.get("type", "")

        print(f"Processing instance {instance_id} (type={instance_type})...", end="", flush=True)

        try:
            if instance_type == "direct":
                judge_result = await judge_direct_with_policy(client, model, instance)
            elif instance_type == "indirect":
                judge_result = await judge_indirect_with_policy(client, model, instance)
            else:
                judge_result = {"error": f"Unknown type: {instance_type}"}

            print(" done", flush=True)
            return {**instance, "judge_result": judge_result}

        except Exception as e:
            print(f" error: {e}", flush=True)
            return {**instance, "judge_result": {"error": str(e)}}


def load_instances(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = json.loads(input_path.read_text(encoding="utf-8"))

    if not isinstance(raw, list):
        raise ValueError(f"Input file must contain a JSON array, got {type(raw).__name__}")

    return raw


def load_existing_results(output_path: Path) -> List[Dict[str, Any]]:
    if not output_path.exists():
        return []

    try:
        raw = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            print(f"Loaded {len(raw)} existing results from {output_path}")
            return raw
        return []
    except Exception as e:
        print(f"[WARN] Failed to load existing results: {e}")
        return []


def get_instance_id(instance: Dict[str, Any]) -> str:
    doc_id = instance.get("doc_id", "")
    query = instance.get("query", "")
    return f"{doc_id}::{query}"


def save_results_incremental(output_path: Path, results: List[Dict[str, Any]]) -> None:
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


async def main_async(args: argparse.Namespace) -> None:
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / args.output_file

    instances = load_instances(input_path)
    print(f"Loaded {len(instances)} instances from {input_path}")

    existing_results = load_existing_results(out_path)
    processed_ids = {get_instance_id(r) for r in existing_results}

    pending_instances = [
        inst for inst in instances
        if get_instance_id(inst) not in processed_ids
    ]

    print(f"Already processed: {len(processed_ids)}, Pending: {len(pending_instances)}")

    if not pending_instances:
        print("All instances already processed. Nothing to do.")
        input_stem = input_path.stem
        metrics_filename = f"{input_stem}_metrics.json"
        print_summary(existing_results, output_dir, metrics_filename)
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    sem = asyncio.Semaphore(max(1, args.batch_size))

    save_lock = asyncio.Lock()
    all_results: List[Dict[str, Any]] = list(existing_results)

    async def process_and_save(instance: Dict[str, Any]) -> Dict[str, Any]:
        result = await process_instance(sem, client, args.model, instance)

        async with save_lock:
            all_results.append(result)
            await asyncio.to_thread(
                save_results_incremental, out_path, all_results
            )

        return result

    tasks = [
        asyncio.create_task(process_and_save(instance))
        for instance in pending_instances
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures = [r for r in results if isinstance(r, Exception)]

    if failures:
        print(f"\n[WARN] {len(failures)} task(s) failed.")
        for e in failures[:10]:
            print(f" - {type(e).__name__}: {e}")

    print(f"\nTotal items saved: {len(all_results)} -> {out_path}")

    input_stem = input_path.stem
    metrics_filename = f"{input_stem}_metrics.json"
    print_summary(all_results, output_dir, metrics_filename)


def print_summary(items: List[Dict[str, Any]], output_dir: Path, metrics_filename: str = "summary_metrics.json") -> None:
    print("\n" + "="*60)
    print("EVALUATION SUMMARY (DVA Mitigation)")
    print("="*60)

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        judge_result = item.get("judge_result", {})
        judge_type = judge_result.get("judge_type", "unknown")
        if judge_type not in by_type:
            by_type[judge_type] = []
        by_type[judge_type].append(item)

    summary_data: Dict[str, Any] = {}

    for judge_type, type_items in sorted(by_type.items()):
        print(f"\n[{judge_type}] ({len(type_items)} items)")

        type_summary: Dict[str, Any] = {
            "count": len(type_items)
        }

        if judge_type == "direct_with_policy":
            leaked_count = sum(
                1 for item in type_items
                if item.get("judge_result", {}).get("leakage", {}).get("leaked") is True
            )
            leakage_rate = leaked_count / len(type_items) if len(type_items) > 0 else 0.0
            print(f"  Leakage rate: {leaked_count}/{len(type_items)} ({leakage_rate*100:.1f}%)")
            type_summary["leakage_count"] = leaked_count
            type_summary["leakage_rate"] = leakage_rate

        elif judge_type == "indirect_with_policy":
            leaked_count = sum(
                1 for item in type_items
                if item.get("judge_result", {}).get("leakage", {}).get("leaked") is True
            )
            leakage_rate = leaked_count / len(type_items) if len(type_items) > 0 else 0.0
            print(f"  Leakage rate: {leaked_count}/{len(type_items)} ({leakage_rate*100:.1f}%)")
            type_summary["leakage_count"] = leaked_count
            type_summary["leakage_rate"] = leakage_rate

            faith_rates = [
                item.get("judge_result", {}).get("faithfulness", {}).get("rate")
                for item in type_items
            ]
            valid_rates = [r for r in faith_rates if r is not None]
            if valid_rates:
                avg_rate = sum(valid_rates) / len(valid_rates)
                print(f"  Avg faithfulness rate: {avg_rate*100:.1f}%")
                type_summary["avg_faithfulness_rate"] = avg_rate
            else:
                type_summary["avg_faithfulness_rate"] = None

        summary_data[judge_type] = type_summary

    metrics_dir = METRICS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / metrics_filename

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, ensure_ascii=False, indent=2, fp=f)

    print(f"\n{'='*60}")
    print(f"Summary metrics saved to: {metrics_path}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation for DVA mitigation responses (uses final_answer)."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input JSON file (DVA evaluation output from 04_mitigation_dva.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to save output JSON file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="judge_dva_output.json",
        help="Output JSON filename.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("JUDGE_MODEL", "openai/gpt-4o"),
        help="Judge model id (e.g., openai/gpt-4o).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum number of concurrent API calls.",
    )

    args = parser.parse_args()

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("Mitigation: DVA - using final_answer for evaluation")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
