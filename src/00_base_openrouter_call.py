from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI

try:
    from prompt import SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE
except ImportError:
    SYSTEM_PROMPT_TEMPLATE = None
    USER_PROMPT_TEMPLATE = None


# Path configuration
ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs_clip"
OUTPUT_DIR = ROOT / "data"


def load_instances(input_path: Path) -> List[Dict[str, Any]]:
    """Read the input JSON file and return a list of instances."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse input file {input_path}: {exc}")

    if not isinstance(raw, list):
        raise ValueError(f"Input file must contain a JSON array, got {type(raw).__name__}")

    return raw


def formulate_system_prompt(instance: Dict[str, Any]) -> str:
    """
    Build the system prompt from the provided instance.

    Args:
        instance: Input data that may include doc_id, policy, policy_value, etc.

    Returns:
        The system prompt string
    """
    if SYSTEM_PROMPT_TEMPLATE:
        return SYSTEM_PROMPT_TEMPLATE.format(**instance)

    # Default system prompt
    return "You are a helpful assistant that processes documents and generates structured responses."


def formulate_user_prompt(instance: Dict[str, Any]) -> str:
    """
    Build the user prompt from the provided instance.

    Args:
        instance: Input data that may include doc_id, question, policy, policy_value, etc.

    Returns:
        A user prompt string or JSON payload
    """
    if USER_PROMPT_TEMPLATE:
        return USER_PROMPT_TEMPLATE.format(**instance)

    # Default: serialize the instance to JSON
    return json.dumps(instance, ensure_ascii=False, separators=(",", ":"))


def process_model_output(llm_text: str) -> Dict[str, Any]:
    """
    Parse the LLM response and return structured data.

    Args:
        llm_text: Raw response text from the LLM

    Returns:
        Parsed data as a dict, or an empty dict on failure
    """
    text = (llm_text or "").strip()
    if not text:
        return {}

    # Attempt to parse JSON directly
    data = None
    try:
        data = json.loads(text)
    except Exception:
        # Fallback: attempt to extract a JSON array
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                pass
        # Fallback: attempt to extract a JSON object
        if data is None:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:
                    pass

    if data is None:
        return {}

    # If we got a list, return the first dict entry
    if isinstance(data, list):
        return data[0] if data and isinstance(data[0], dict) else {}

    # Dicts are returned as-is
    if isinstance(data, dict):
        return data

    return {}


async def process_instance(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    pdf_engine: str,
    instance: Dict[str, Any],
    docs_dir: Path,
) -> Dict[str, Any]:
    """
    Process a single instance by calling the LLM and returning the result.

    Args:
        sem: Semaphore limiting concurrency
        client: OpenAI API client
        model: Target model identifier
        pdf_engine: PDF parsing engine identifier
        instance: Input row (must contain doc_id)
        docs_dir: Directory containing PDF files

    Returns:
        The merged instance plus raw_output and parsed_output fields
    """
    async with sem:
        doc_id = instance.get("doc_id", "")
        if not doc_id:
            print(f"[WARN] instance missing doc_id: {instance}")
            return {**instance, "raw_output": "", "parsed_output": {}}

        print(f"Processing {doc_id} ...", end="", flush=True)

        # PDF path
        pdf_path = docs_dir / doc_id
        if not pdf_path.exists():
            print(f" [ERROR: PDF not found] ", flush=True)
            return {
                **instance,
                "raw_output": "",
                "parsed_output": {},
                "error": f"PDF file not found: {pdf_path}",
            }

        # Base64 encode the PDF for API upload
        pdf_bytes = await asyncio.to_thread(pdf_path.read_bytes)
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        data_url = f"data:application/pdf;base64,{b64}"

        # Build the prompts
        system_prompt = formulate_system_prompt(instance)
        user_prompt_text = formulate_user_prompt(instance)

        # Call the OpenRouter API
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
        parsed_output = process_model_output(raw_output)

        print(" done", flush=True)
        return {
            **instance,
            "raw_output": raw_output,
            "parsed_output": parsed_output,
        }


async def main_async(args: argparse.Namespace) -> None:
    """Async entry point."""
    docs_dir = Path(args.docs_dir)
    output_dir = Path(args.output_dir)
    input_file = Path(args.input_file)

    # Load instances
    instances = load_instances(input_file)
    if not instances:
        raise SystemExit(f"No instances found in {input_file}")

    print(f"Loaded {len(instances)} instances from {input_file}")

    # Create the OpenAI client
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # Limit concurrency with a semaphore (batch_size)
    sem = asyncio.Semaphore(max(1, args.batch_size))

    # Process every instance
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

    # Collect successful vs failed tasks
    final_items: List[Dict[str, Any]] = []
    failures: List[Exception] = []

    for r in results:
        if isinstance(r, Exception):
            failures.append(r)
        else:
            final_items.append(r)

    # Persist results to disk
    out_path = output_dir / args.output_file
    await asyncio.to_thread(
        out_path.write_text,
        json.dumps(final_items, ensure_ascii=False, indent=2),
        "utf-8",
    )

    if failures:
        print(f"\n[WARN] {len(failures)} task(s) failed.")
        for e in failures[:10]:
            print(f" - {type(e).__name__}: {e}")

    print(f"\nWrote {len(final_items)} items -> {out_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process instances with PDFs using OpenRouter API."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input JSON file containing instances (e.g., data/00_original_queries.json).",
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
        help="Directory to save output JSON file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output.json",
        help="Output JSON filename.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "openai/gpt-5.2"),
        help="OpenRouter model id (e.g., openai/gpt-5.2).",
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
