# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import sys
import time
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src folder to path to enable importing tpm_mlx modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tpm_mlx.engine import MLXEngine

# --- TEST SUITE DEFINITIONS ---

def generate_long_context_haystack() -> Tuple[str, Dict[str, str]]:
    """Generates a ~8,000+ token context haystack with embedded needles."""
    needles = {
        "hex_code": "0x9F4B12A8",
        "secret_token": "SEC_9921_ALPHA",
        "target_uuid": "e4a8b192-33c9-4b11-a8df-7210948ac912"
    }
    
    filler_lines = [
        "[2026-07-26 08:00:01] INFO: System status nominal. Memory usage at 34%.",
        "[2026-07-26 08:00:15] DEBUG: Cache rebalancing event completed in 1.4ms.",
        "[2026-07-26 08:01:00] INFO: Periodic health check passed on node cluster alpha.",
        "[2026-07-26 08:02:44] DEBUG: Incoming connection accepted from 192.168.1.104.",
        "[2026-07-26 08:03:12] INFO: Garbage collection sweep freed 142MB RAM."
    ]
    
    # Expand to ~3000 lines of logs to simulate large context window
    logs = []
    for i in range(500):
        logs.extend(filler_lines)
        if i == 120:
            logs.append(f"[2026-07-26 08:10:45] CRITICAL_AUDIT_EVENT: Hex error code logged: {needles['hex_code']}. Intervention required.")
        if i == 280:
            logs.append(f"[2026-07-26 08:25:11] SECURITY_NOTICE: Session auth token issued: {needles['secret_token']} for admin operation.")
        if i == 410:
            logs.append(f"[2026-07-26 08:40:02] DATABASE_WARN: Transaction failed for UUID: {needles['target_uuid']}. Rollback triggered.")

    return "\n".join(logs), needles


QUALITY_TESTS = [
    {
        "id": "json_schema_constraint",
        "name": "1. Strict JSON Schema & Negative Constraint",
        "description": "Evaluates schema accuracy and adherence to negative constraints (no markdown fences).",
        "prompt": (
            "Generate a user profile object strictly adhering to this JSON schema:\n"
            "{\n"
            "  \"type\": \"object\",\n"
            "  \"properties\": {\n"
            "    \"user_id\": {\"type\": \"integer\"},\n"
            "    \"username\": {\"type\": \"string\"},\n"
            "    \"role\": {\"type\": \"string\", \"enum\": [\"admin\", \"editor\", \"viewer\"]},\n"
            "    \"permissions\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}}\n"
            "  },\n"
            "  \"required\": [\"user_id\", \"username\", \"role\", \"permissions\"]\n"
            "}\n"
            "CRITICAL REQUIREMENT: Output raw JSON ONLY. Do NOT wrap in markdown code fences (```json or ```)."
        ),
        "validator": "validate_json_constraint"
    },
    {
        "id": "needle_in_haystack",
        "name": "2. Multi-Needle Extraction (Long Context)",
        "description": "Tests retrieval precision for 3 hidden keys in a long context window.",
        "prompt_builder": lambda haystack: (
            f"Below is a long system log archive:\n\n{haystack}\n\n"
            "Task: Locate and extract the following three specific values from the log archive:\n"
            "1. The CRITICAL_AUDIT_EVENT Hex error code\n"
            "2. The SECURITY_NOTICE auth token\n"
            "3. The DATABASE_WARN Transaction UUID\n\n"
            "Provide the extracted values clearly."
        ),
        "validator": "validate_needle_extraction"
    },
    {
        "id": "multi_constraint_ifeval",
        "name": "3. Multi-Constraint Instruction Adherence (IFEval Style)",
        "description": "Tests compliance with 4 simultaneous structural and negative formatting rules.",
        "prompt": (
            "Write a project status update for Project Titan based on these strict constraints:\n"
            "1. You must write EXACTLY 4 bullet points (starting with '* ' or '- ').\n"
            "2. You MUST include the exact word 'COMPLETED' in ALL CAPS.\n"
            "3. You must NOT use the words 'problem', 'issue', or 'delay' anywhere in your response.\n"
            "4. Your total response must be under 120 words."
        ),
        "validator": "validate_ifeval_constraints"
    },
    {
        "id": "data_synthesis_nuance",
        "name": "4. Deep Data Synthesis & Fact Recall",
        "description": "Tests accuracy in synthesizing key quantitative facts from a multi-project report.",
        "prompt": (
            "Read the following project audit report:\n"
            "Project Alpha reached $1.4M Q2 revenue with 42% YoY growth, despite 3 critical security vulnerabilities identified. "
            "Project Beta suffered a 6-week delay due to supply chain bottlenecks, but secured $800K in enterprise pre-orders. "
            "Project Gamma achieved a 99.99% uptime SLA across 12 regions.\n\n"
            "Synthesize an executive summary highlighting key figures for all three projects."
        ),
        "validator": "validate_data_synthesis"
    }
]


# --- VALIDATION FUNCTIONS ---

def validate_json_constraint(response: str) -> Dict[str, Any]:
    raw = response.strip()
    checks = {
        "valid_json": False,
        "schema_matched": False,
        "no_markdown_fence": not raw.startswith("```") and not raw.endswith("```")
    }
    
    # Try parsing JSON
    try:
        # Strip potential markdown fences just for JSON check if model failed fence constraint
        clean_json = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(clean_json)
        checks["valid_json"] = True
        
        if isinstance(data, dict):
            req_keys = {"user_id", "username", "role", "permissions"}
            if req_keys.issubset(data.keys()):
                if isinstance(data["user_id"], int) and isinstance(data["username"], str) and data["role"] in ["admin", "editor", "viewer"] and isinstance(data["permissions"], list):
                    checks["schema_matched"] = True
    except Exception:
        pass
        
    score = (sum(1 for v in checks.values() if v) / len(checks)) * 100
    return {"score": score, "details": checks}


def validate_needle_extraction(response: str, needles: Dict[str, str]) -> Dict[str, Any]:
    checks = {
        "hex_code_found": needles["hex_code"] in response,
        "secret_token_found": needles["secret_token"] in response,
        "target_uuid_found": needles["target_uuid"] in response
    }
    score = (sum(1 for v in checks.values() if v) / len(checks)) * 100
    return {"score": score, "details": checks}


def validate_ifeval_constraints(response: str) -> Dict[str, Any]:
    text = response.strip()
    words = text.split()
    bullets = [line for line in text.split("\n") if line.strip().startswith("*") or line.strip().startswith("-")]
    
    checks = {
        "exactly_4_bullets": len(bullets) == 4,
        "contains_COMPLETED_uppercase": "COMPLETED" in text,
        "no_forbidden_words": not any(w in text.lower() for w in ["problem", "issue", "delay"]),
        "under_120_words": len(words) <= 120
    }
    score = (sum(1 for v in checks.values() if v) / len(checks)) * 100
    return {"score": score, "details": checks}


def validate_data_synthesis(response: str) -> Dict[str, Any]:
    text = response.lower()
    facts = {
        "alpha_revenue": "$1.4m" in text or "1.4 million" in text or "1.4m" in text,
        "alpha_growth": "42%" in text,
        "beta_orders": "$800k" in text or "800,000" in text or "800k" in text,
        "gamma_sla": "99.99%" in text
    }
    score = (sum(1 for v in facts.values() if v) / len(facts)) * 100
    return {"score": score, "details": facts}


# --- HARNESS EXECUTION ---

def evaluate_model(model_path: str, max_kv_size: int = 8192) -> Dict[str, Any]:
    """Runs all quality benchmark tests on a specified model."""
    print(f"\n==================================================")
    print(f"  Evaluating Quality: {model_path}")
    print(f"==================================================")
    
    engine = MLXEngine(model_path_or_id=model_path, max_kv_size=max_kv_size)
    haystack_text, needles = generate_long_context_haystack()
    
    results = []
    total_score = 0.0
    
    for test in QUALITY_TESTS:
        print(f"\n[Running Test] {test['name']}...")
        
        # Build prompt
        if test["id"] == "needle_in_haystack":
            prompt_text = test["prompt_builder"](haystack_text)
        else:
            prompt_text = test["prompt"]
            
        formatted_prompt = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate model output (with reasoning disabled for clean extraction)
        start_t = time.perf_counter()
        raw_outputs = list(engine.generate_stream(
            prompt=formatted_prompt,
            max_tokens=2048,
            temperature=0.0,
            show_reasoning=False
        ))
        dur = time.perf_counter() - start_t
        
        full_text = "".join(r.text for r in raw_outputs).strip()
        
        # Run evaluation validator
        validator_type = test["validator"]
        if validator_type == "validate_json_constraint":
            eval_res = validate_json_constraint(full_text)
        elif validator_type == "validate_needle_extraction":
            eval_res = validate_needle_extraction(full_text, needles)
        elif validator_type == "validate_ifeval_constraints":
            eval_res = validate_ifeval_constraints(full_text)
        elif validator_type == "validate_data_synthesis":
            eval_res = validate_data_synthesis(full_text)
        else:
            eval_res = {"score": 0.0, "details": {}}
            
        score = eval_res["score"]
        total_score += score
        print(f"  --> Score: {score:.1f}% (Execution Time: {dur:.2f}s)")
        print(f"      Details: {eval_res['details']}")
        
        results.append({
            "test_id": test["id"],
            "name": test["name"],
            "score": score,
            "details": eval_res["details"],
            "execution_time_s": dur,
            "response_preview": full_text[:150].replace("\n", " ") + "..."
        })
        
    avg_score = total_score / len(QUALITY_TESTS)
    print(f"\n---> Overall Model Quality Score: {avg_score:.1f}%\n")
    
    return {
        "model": model_path,
        "overall_score": avg_score,
        "tests": results
    }


def main():
    parser = argparse.ArgumentParser(description="TPM-MLX Quality & Accuracy Benchmark Harness")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "mlx-community/gemma-4-e4b-it-4bit",
            "mlx-community/gemma-4-12B-it-4bit",
            "mlx-community/gemma-4-12B-it-qat-4bit"
        ],
        help="List of model HuggingFace IDs or paths to benchmark"
    )
    parser.add_argument("--max-kv-size", type=int, default=8192, help="Pre-allocated KV Cache Size")
    args = parser.parse_args()

    all_results = []
    for model_path in args.models:
        try:
            res = evaluate_model(model_path, max_kv_size=args.max_kv_size)
            all_results.append(res)
        except Exception as e:
            print(f"ERROR evaluating {model_path}: {e}")

    # Generate Markdown Summary Report
    report_lines = [
        "# TPM-MLX Model Quality & Accuracy Scorecard",
        "",
        "Evaluation comparing model accuracy across **Instruction Following**, **JSON Schema Adherence**, **Needle Retrieval**, and **Data Synthesis**.",
        "",
        "| Model | Overall Quality Score | JSON Constraint % | Needle Retrieval % | Multi-Rule Adherence % | Synthesis Accuracy % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for r in all_results:
        model_name = r["model"]
        overall = f"{r['overall_score']:.1f}%"
        test_scores = {t["test_id"]: f"{t['score']:.1f}%" for t in r["tests"]}
        report_lines.append(
            f"| `{model_name}` | **{overall}** | "
            f"{test_scores.get('json_schema_constraint', 'N/A')} | "
            f"{test_scores.get('needle_in_haystack', 'N/A')} | "
            f"{test_scores.get('multi_constraint_ifeval', 'N/A')} | "
            f"{test_scores.get('data_synthesis_nuance', 'N/A')} |"
        )

    report_lines.extend([
        "",
        "## Methodology Notes",
        "- **JSON Constraint**: Validates strict JSON syntax, target field mapping, and negative constraint adherence (no markdown code blocks).",
        "- **Needle Retrieval**: Tests 3 exact hex/UUID needles placed in a 8,000+ token log archive.",
        "- **Multi-Rule Adherence**: Evaluates compliance with 4 simultaneous negative and structural formatting rules (IFEval style).",
        "- **Data Synthesis**: Measures factual extraction recall from multi-project quantitative reports."
    ])

    report_path = Path(__file__).parent / "quality_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n==================================================")
    print(f"Quality Benchmark Complete! Report saved to {report_path}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
