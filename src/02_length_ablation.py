from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import sys

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

from openai import AsyncOpenAI

try:
    from prompts.evaluate_model import (
        SYSTEM_PROMPT_TEMPLATE_WITH_POLICY,
        SYSTEM_PROMPT_TEMPLATE_WITHOUT_POLICY,
    )
except ImportError:
    SYSTEM_PROMPT_TEMPLATE_WITH_POLICY = None
    SYSTEM_PROMPT_TEMPLATE_WITHOUT_POLICY = None

# Path configuration
ROOT = Path(__file__).resolve().parents[1]
ABLATION_DOCS_BASE_DIR = ROOT / "data" / "context_len_ablation_doc_clips"
OUTPUT_DIR = ROOT / "results"

VALID_CONTEXT_LENGTHS = [3, 5, 7, 9, 11]


def pdf_to_base64_images(pdf_path: Path, max_pages: int, resolution: int) -> List[str]:
    """Convert PDF pages to base64-encoded PNG data URLs."""
    if fitz is None:
        raise RuntimeError("PyMuPDF is required when using --doc-mode image")

    encoded_images: List[str] = []
    page_limit = max_pages if max_pages > 0 else None
    with fitz.open(pdf_path) as pdf:
        for index, page in enumerate(pdf):
            if page_limit is not None and index >= page_limit:
                break
            pixmap = page.get_pixmap(dpi=resolution)
            encoded = base64.b64encode(pixmap.tobytes("png")).decode("utf-8")
            encoded_images.append(encoded)
    return encoded_images


def load_instances(input_path: Path) -> List[Dict[str, Any]]:
    """Read the input JSON file and return the list of instances."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse input file {input_path}: {exc}")

    if not isinstance(raw, list):
        raise ValueError(f"Input file must contain a JSON array, got {type(raw).__name__}")

    return raw


def load_existing_results(output_path: Path) -> List[Dict[str, Any]]:
    """Load previously saved results if the file exists (partial files allowed)."""
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
    """Return the unique identifier composed of doc_id and query."""
    doc_id = instance.get("doc_id", "")
    query = instance.get("query", "")
    return f"{doc_id}::{query}"


def save_results_incremental(output_path: Path, results: List[Dict[str, Any]]) -> None:
    """Persist the merged (existing + new) results to disk."""
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def formulate_system_prompt(instance: Dict[str, Any], with_policy: bool) -> str:
    """
    Build the system prompt from the given instance.

    Args:
        instance: Input record that may include the policy text
        with_policy: Whether to include the policy prompt

    Returns:
        The fully formatted system prompt
    """
    if with_policy:
        if not SYSTEM_PROMPT_TEMPLATE_WITH_POLICY:
            raise ValueError("SYSTEM_PROMPT_TEMPLATE_WITH_POLICY is not defined.")
        policy = instance.get("policy", "")
        return SYSTEM_PROMPT_TEMPLATE_WITH_POLICY.format(policy=policy)
    else:
        if not SYSTEM_PROMPT_TEMPLATE_WITHOUT_POLICY:
            raise ValueError("SYSTEM_PROMPT_TEMPLATE_WITHOUT_POLICY is not defined.")
        return SYSTEM_PROMPT_TEMPLATE_WITHOUT_POLICY


def formulate_user_prompt(instance: Dict[str, Any]) -> str:
    """
    Build the user prompt from the given instance.

    Args:
        instance: Input record containing the query

    Returns:
        The user prompt string
    """
    return instance.get("query", "")


async def process_instance(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    pdf_engine: str,
    doc_mode: str,
    max_pages: int,
    image_resolution: int,
    instance: Dict[str, Any],
    docs_dir: Path,
    with_policy: bool,
    enable_thinking: bool,
    context_length: int,
) -> Dict[str, Any]:
    """
    Process a single instance and return the LLM response.

    Args:
        sem: Concurrency limiter
        client: OpenAI API client
        model: Model identifier
        pdf_engine: PDF parsing backend
        doc_mode: 'pdf' uploads the file, 'image' includes page rasters
        max_pages: Maximum page count when rasterizing
        image_resolution: DPI to use when rasterizing
        instance: Input record (requires doc_id)
        docs_dir: Directory that stores the source PDFs
        with_policy: Whether to use the policy-aware prompt
        enable_thinking: Whether to enable reasoning/thinking mode
        context_length: Context length bucket being evaluated

    Returns:
        The merged instance plus model metadata and response
    """
    async with sem:
        doc_id = instance.get("doc_id", "")
        if not doc_id:
            print(f"[WARN] instance missing doc_id: {instance}")
            return {**instance, "model_response": "", "error": "Missing doc_id"}

        print(f"Processing {doc_id} (context_length={context_length}) ...", end="", flush=True)

        # Resolve the PDF path for this doc_id
        pdf_path = docs_dir / doc_id
        if not pdf_path.exists():
            print(f" [ERROR: PDF not found] ", flush=True)
            return {
                **instance,
                "model_response": "",
                "error": f"PDF file not found: {pdf_path}",
            }

        # Build both prompts
        system_prompt = formulate_system_prompt(instance, with_policy)
        user_prompt_text = formulate_user_prompt(instance)
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": user_prompt_text}
        ]
        extra_body: Dict[str, Any] | None = None

        if doc_mode == "pdf":
            pdf_bytes = await asyncio.to_thread(pdf_path.read_bytes)
            b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            data_url = f"data:application/pdf;base64,{b64}"
            user_content.append(
                {
                    "type": "file",
                    "file": {"filename": doc_id, "file_data": data_url},
                }
            )
            extra_body = {"plugins": [{"id": "file-parser", "pdf": {"engine": pdf_engine}}]}
        else:
            try:
                encoded_images = await asyncio.to_thread(
                    pdf_to_base64_images,
                    pdf_path,
                    max_pages,
                    image_resolution,
                )
            except Exception as exc:
                print(f" [ERROR converting {doc_id}: {exc}] ", flush=True)
                return {
                    **instance,
                    "model_response": "",
                    "error": f"Failed to render PDF as images: {exc}",
                }

            if not encoded_images:
                print(" [ERROR: No images generated] ", flush=True)
                return {
                    **instance,
                    "model_response": "",
                    "error": "No page images generated from PDF.",
                }

            for encoded in encoded_images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    }
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
        }

        # Attach PDF engine + reasoning settings to extra_body
        combined_extra_body = extra_body.copy() if extra_body else {}
        combined_extra_body["reasoning"] = {"enabled": enable_thinking}
        request_kwargs["extra_body"] = combined_extra_body

        completion = await client.chat.completions.create(**request_kwargs)

        model_response = completion.choices[0].message.content or ""

        print(" done", flush=True)
        return {
            **instance,
            "context_length": context_length,
            "doc_mode": doc_mode,
            "with_policy": with_policy,
            "pdf_engine": pdf_engine,
            "enable_thinking": enable_thinking,
            "model": model,
            "model_response": model_response,
        }


async def main_async(args: argparse.Namespace) -> None:
    """Async entry point for the context-length ablation."""
    # Select the docs directory based on context_length
    docs_dir = ABLATION_DOCS_BASE_DIR / str(args.context_length)
    if not docs_dir.exists():
        raise SystemExit(f"Docs directory not found: {docs_dir}")

    output_dir = Path(args.output_dir)
    input_file = Path(args.input_file)

    # Auto-generate the output filename
    model_name = args.model.replace("/", "_").replace(":", "_")
    policy_status = "with_policy" if args.with_policy else "without_policy"
    output_filename = f"context_len_{args.context_length}_{args.doc_mode}_{policy_status}_{model_name}.json"
    out_path = output_dir / output_filename

    # Load all instances (dataset already restricted to indirect queries)
    instances = load_instances(input_file)

    if not instances:
        raise SystemExit(f"No instances found in {input_file}")

    print(f"Loaded {len(instances)} instances from {input_file}")
    print(f"Using docs from: {docs_dir}")

    # Load prior results and skip already-processed items
    existing_results = load_existing_results(out_path)
    processed_ids = {get_instance_id(r) for r in existing_results}

    # Keep only the pending instances
    pending_instances = [
        inst for inst in instances
        if get_instance_id(inst) not in processed_ids
    ]

    print(f"Already processed: {len(processed_ids)}, Pending: {len(pending_instances)}")

    if not pending_instances:
        print("All instances already processed. Nothing to do.")
        return

    # Create the OpenAI client
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # Limit concurrency via a semaphore (batch_size)
    sem = asyncio.Semaphore(max(1, args.batch_size))

    # Lock + buffer to support incremental saving
    save_lock = asyncio.Lock()
    all_results: List[Dict[str, Any]] = list(existing_results)

    async def process_and_save(instance: Dict[str, Any]) -> Dict[str, Any]:
        """Process an instance and immediately flush it to disk."""
        result = await process_instance(
            sem=sem,
            client=client,
            model=args.model,
            pdf_engine=args.pdf_engine,
            doc_mode=args.doc_mode,
            max_pages=args.max_pages,
            image_resolution=args.image_resolution,
            instance=instance,
            docs_dir=docs_dir,
            with_policy=args.with_policy,
            enable_thinking=args.enable_thinking,
            context_length=args.context_length,
        )

        # Append while holding the lock to prevent concurrent writes
        async with save_lock:
            all_results.append(result)
            await asyncio.to_thread(
                save_results_incremental, out_path, all_results
            )

        return result

    # Kick off all pending tasks
    tasks = [
        asyncio.create_task(process_and_save(instance))
        for instance in pending_instances
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Summarize failures if any
    failures = [r for r in results if isinstance(r, Exception)]

    if failures:
        print(f"\n[WARN] {len(failures)} task(s) failed.")
        for e in failures[:10]:
            print(f" - {type(e).__name__}: {e}")

    print(f"\nTotal items saved: {len(all_results)} -> {out_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Context length ablation study for indirect queries using OpenRouter API."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(ROOT / "data" / "03_length_ablation_data.json"),
        help="Path to input JSON file (default: data/03_length_ablation_data.json).",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        required=True,
        choices=VALID_CONTEXT_LENGTHS,
        help=f"Context length to use. Must be one of: {VALID_CONTEXT_LENGTHS}",
    )
    parser.add_argument(
        "--with-policy",
        action="store_true",
        help="If set, use system prompt with policy. Otherwise, use without policy.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to save output JSON file.",
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
        "--doc-mode",
        type=str,
        default="pdf",
        choices=["pdf", "image"],
        help="Handle documents as raw PDFs or rasterized images.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=120,
        help="Maximum number of pages to rasterize when --doc-mode image is used.",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=144,
        help="DPI for rasterizing PDFs when --doc-mode image is selected.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Maximum number of concurrent API calls.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="If set, enable reasoning/thinking mode in the API request.",
    )

    args = parser.parse_args()
    # print all args
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"Docs directory: {ABLATION_DOCS_BASE_DIR / str(args.context_length)}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
