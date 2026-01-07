from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI

PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

try:
    from prompts.check_initial_faithfulness_checklists import (
        SYSTEM_PROMPT_TEMPLATE,
        USER_PROMPT_TEMPLATE,
    )
except ImportError:
    SYSTEM_PROMPT_TEMPLATE = None
    USER_PROMPT_TEMPLATE = None


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs_clip"
OUTPUT_DIR = ROOT / "data"
DEFAULT_INPUT_FILE = ROOT / "data" / "initial_faithfulness_checklists.json"
DEFAULT_OUTPUT_FILE = "check_initial_faithfulness_checklists.json"

CRITERIA_LINE_PATTERN = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
IMPORTANT_FIELDS = ["id", "doc_id", "question", "policy", "policy_value"]


def load_instances(input_path: Path) -> List[Dict[str, Any]]:
    """Read the input JSON file and return the list of instances."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse input file {input_path}: {exc}")

    if not isinstance(raw, list):
        raise ValueError(
            f"Input file must contain a JSON array, got {type(raw).__name__}"
        )

    return raw


def format_criteria_list(criteria: Any) -> str:
    """Convert the parsed_output list into a numbered string."""
    if not isinstance(criteria, list):
        return ""

    lines: List[str] = []
    for idx, item in enumerate(criteria, start=1):
        text = str(item).strip()
        lines.append(f"{idx}. {text}")

    return "\n".join(lines)


def formulate_system_prompt() -> str:
    if SYSTEM_PROMPT_TEMPLATE:
        return SYSTEM_PROMPT_TEMPLATE

    return (
        "You are a helpful assistant that validates checklist items using a PDF and a policy."
    )


def instance_descriptor(instance: Dict[str, Any]) -> str:
    question = str(instance.get("question", "")).replace("\n", " ")
    return f"id={instance.get('id', '?')}, question[:20]={question[:20]}"


def prepare_result(instance: Dict[str, Any]) -> Dict[str, Any]:
    result = {field: instance.get(field) for field in IMPORTANT_FIELDS}
    result["initial_checklist"] = instance.get("parsed_output", [])
    result["check_raw_output"] = ""
    result["check_parsed_output"] = []
    result["error"] = ""
    return result


def formulate_user_prompt(instance: Dict[str, Any]) -> str:
    if not USER_PROMPT_TEMPLATE:
        raise ValueError("USER_PROMPT_TEMPLATE is not defined.")

    criteria_text = format_criteria_list(instance.get("parsed_output"))
    if not criteria_text:
        raise ValueError(
            f"Instance {instance.get('doc_id', 'unknown')} has no parsed criteria."
        )

    question = instance.get("question", "")
    policy = str(instance.get("policy", ""))
    policy_value = str(instance.get("policy_value", ""))

    return USER_PROMPT_TEMPLATE.format(
        question=question,
        criteria=criteria_text,
        policy_target=f'"{policy}"',
        policy_value=f'"{policy_value}"',
    )


def process_model_output(llm_text: str, instance: Dict[str, Any]) -> List[str]:
    """Parse the LLM response into a list of per-item judgments."""
    text = (llm_text or "").strip()
    if not text:
        return []

    parsed: List[str] = []
    for line in text.splitlines():
        match = CRITERIA_LINE_PATTERN.match(line)
        if match:
            parsed.append(match.group(2).strip())

    if parsed:
        return parsed

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item).strip() for item in data]
    except Exception:
        pass

    print(
        f"[WARN parsing failed {instance_descriptor(instance)}]",
        flush=True,
    )
    return [text]


async def process_instance(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    pdf_engine: str,
    instance: Dict[str, Any],
    docs_dir: Path,
) -> Dict[str, Any]:
    async with sem:
        result = prepare_result(instance)
        doc_id = result.get("doc_id") or ""
        if not doc_id:
            desc = instance_descriptor(instance)
            message = "Missing doc_id"
            print(f"[ERROR {desc}] {message}", flush=True)
            result["error"] = message
            return result

        print(f"Checking {doc_id} ...", end="", flush=True)

        pdf_path = docs_dir / doc_id
        if not pdf_path.exists():
            message = f"PDF file not found: {pdf_path}"
            print(
                f" [ERROR {instance_descriptor(instance)}] {message}",
                flush=True,
            )
            result["error"] = message
            return result

        try:
            user_prompt_text = formulate_user_prompt(instance)
        except ValueError as exc:
            print(
                f" [ERROR {instance_descriptor(instance)}] {exc}",
                flush=True,
            )
            result["error"] = str(exc)
            return result

        pdf_bytes = await asyncio.to_thread(pdf_path.read_bytes)
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        data_url = f"data:application/pdf;base64,{b64}"

        system_prompt = formulate_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt_text},
                    {
                        "type": "file",
                        "file": {"filename": doc_id, "file_data": data_url},
                    },
                ],
            },
        ]

        plugins = [{"id": "file-parser", "pdf": {"engine": pdf_engine}}]

        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            extra_body={"plugins": plugins},
        )

        raw_output = completion.choices[0].message.content or ""
        parsed_output = process_model_output(raw_output, instance)

        print(" done", flush=True)
        result["check_raw_output"] = raw_output
        result["check_parsed_output"] = parsed_output
        return result


async def main_async(args: argparse.Namespace) -> None:
    docs_dir = Path(args.docs_dir)
    output_dir = Path(args.output_dir)
    input_file = Path(args.input_file)

    instances = load_instances(input_file)
    if not instances:
        raise SystemExit(f"No instances found in {input_file}")

    print(f"Loaded {len(instances)} instances from {input_file}")

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(max(1, args.batch_size))

    tasks = [
        asyncio.create_task(
            process_instance(
                sem=sem,
                client=client,
                model=args.model,
                pdf_engine=args.pdf_engine,
                instance=instance,
                docs_dir=docs_dir,
            )
        )
        for instance in instances
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_items: List[Dict[str, Any]] = []
    failures: List[Exception] = []

    for result in results:
        if isinstance(result, Exception):
            failures.append(result)
        else:
            final_items.append(result)

    out_path = output_dir / args.output_file
    await asyncio.to_thread(
        out_path.write_text,
        json.dumps(final_items, ensure_ascii=False, indent=2),
        "utf-8",
    )

    if failures:
        print(f"\n[WARN] {len(failures)} task(s) failed.")
        for error in failures[:10]:
            print(f" - {type(error).__name__}: {error}")

    print(f"\nWrote {len(final_items)} items -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-check initial faithfulness checklists using an OpenRouter model.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(DEFAULT_INPUT_FILE),
        help="Path to the JSON file containing initial faithfulness checklists.",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=str(DOCS_DIR),
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to save the cross-check results.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Filename for the cross-check output JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "openai/gpt-5.1"),
        help="OpenRouter model id (e.g., openai/gpt-5.1).",
    )
    parser.add_argument(
        "--pdf-engine",
        type=str,
        default="native",
        choices=["native", "mistral-ocr", "pdf-text"],
        help="OpenRouter PDF processing engine.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Maximum number of concurrent API calls.",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
