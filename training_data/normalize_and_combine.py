#!/usr/bin/env python
"""Normalize all JSONL files to standardized format and combine into one."""
import json
import random
import os

random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM = (
    "You are BASTION Risk Intelligence \u2014 an autonomous trade management AI. "
    "You monitor live cryptocurrency positions and make execution decisions. "
    "You output structured JSON with action, reasoning, and execution parameters. "
    "PRIORITY ORDER: 1) Hard Stop breach \u2192 EXIT_100_PERCENT_IMMEDIATELY "
    "2) Safety Net break \u2192 EXIT_FULL 3) Guarding Line break \u2192 REDUCE_SIZE or EXIT_FULL "
    "4) Take Profit targets \u2192 TP_PARTIAL or TP_FULL 5) Trailing Stop updates \u2192 TRAIL_STOP "
    "6) Time-based exits \u2192 REDUCE_SIZE. Core philosophy: Exit on STRUCTURE BREAKS, "
    "not arbitrary targets. Let winners run when structure holds. Scale out intelligently "
    "\u2014 decide HOW MUCH to exit based on structure strength, R-multiple, and market context."
)

VALID_ACTIONS = {
    "HOLD", "TP_PARTIAL", "TP_FULL", "MOVE_STOP_TO_BREAKEVEN",
    "TRAIL_STOP", "EXIT_FULL", "REDUCE_SIZE", "ADJUST_STOP",
    "EXIT_100_PERCENT_IMMEDIATELY"
}

def normalize_assistant(raw):
    """Convert any format to standardized nested format."""
    action = raw.get("action", "HOLD")

    # Extract reasoning
    reasoning_raw = raw.get("reasoning", "")
    if isinstance(reasoning_raw, str):
        reasoning_obj = {
            "structure_analysis": reasoning_raw,
            "data_assessment": "",
            "risk_factors": "",
            "exit_logic": raw.get("exit_logic", "")
        }
    elif isinstance(reasoning_raw, dict):
        reasoning_obj = {
            "structure_analysis": reasoning_raw.get("structure_analysis",
                reasoning_raw.get("market_analysis", "")),
            "data_assessment": reasoning_raw.get("data_assessment",
                reasoning_raw.get("risk_calculation", "")),
            "risk_factors": reasoning_raw.get("risk_factors",
                reasoning_raw.get("decision_logic", "")),
            "exit_logic": reasoning_raw.get("exit_logic",
                raw.get("exit_logic", ""))
        }
    else:
        reasoning_obj = {
            "structure_analysis": "",
            "data_assessment": "",
            "risk_factors": "",
            "exit_logic": raw.get("exit_logic", "")
        }

    # Extract execution
    if isinstance(raw.get("execution"), dict):
        exec_obj = {
            "exit_pct": int(raw["execution"].get("exit_pct", 0)),
            "stop_price": raw["execution"].get("stop_price", None),
            "order_type": raw["execution"].get("order_type", "NONE")
        }
    else:
        exit_pct = raw.get("exit_pct", 0)
        exec_obj = {
            "exit_pct": int(exit_pct) if exit_pct is not None else 0,
            "stop_price": raw.get("stop_price", None),
            "order_type": raw.get("order_type", "NONE")
        }

    # Fix stop_price: ensure it's a number or null
    if exec_obj["stop_price"] is not None:
        try:
            exec_obj["stop_price"] = float(exec_obj["stop_price"])
        except (TypeError, ValueError):
            exec_obj["stop_price"] = None

    # Get reason
    reason = raw.get("reason", "")
    if not reason:
        reason = raw.get("exit_logic", reasoning_obj.get("exit_logic", ""))

    return {
        "action": action,
        "urgency": raw.get("urgency", "LOW"),
        "confidence": float(raw.get("confidence", 0.7)),
        "reason": reason,
        "reasoning": reasoning_obj,
        "execution": exec_obj
    }


def main():
    files = [
        "bastion_risk_v2_standardized.jsonl",
        "tp_partial_examples.jsonl",
        "exit_full_examples.jsonl",
        "adjust_stop_examples.jsonl",
        "trail_stop_examples.jsonl",
        "reduce_size_examples.jsonl",
        "exit_emergency_examples.jsonl",
        "move_stop_be_examples.jsonl",
        "tp_full_examples.jsonl",
    ]

    all_examples = []
    errors = 0
    action_counts = {}

    for fname in files:
        fpath = os.path.join(SCRIPT_DIR, fname)
        if not os.path.exists(fpath):
            print(f"SKIP: {fname} not found")
            continue

        file_count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    msgs = example["messages"]

                    # Parse assistant content
                    assistant_raw = json.loads(msgs[2]["content"])
                    assistant_norm = normalize_assistant(assistant_raw)

                    # Validate action
                    if assistant_norm["action"] not in VALID_ACTIONS:
                        print(f"  WARN {fname}:{line_num}: unknown action '{assistant_norm['action']}'")
                        errors += 1
                        continue

                    # Rebuild with standardized system prompt
                    normalized = {
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": msgs[1]["content"]},
                            {"role": "assistant", "content": json.dumps(assistant_norm)}
                        ]
                    }
                    all_examples.append(normalized)
                    action_counts[assistant_norm["action"]] = action_counts.get(assistant_norm["action"], 0) + 1
                    file_count += 1

                except Exception as e:
                    errors += 1
                    print(f"  ERROR {fname}:{line_num}: {e}")

        print(f"  {fname}: {file_count} examples")

    # Shuffle
    random.shuffle(all_examples)

    # Write combined file
    out_path = os.path.join(SCRIPT_DIR, "bastion_risk_v2_combined.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"NORMALIZATION COMPLETE")
    print(f"{'='*50}")
    print(f"Total examples: {len(all_examples)}")
    print(f"Errors: {errors}")
    print(f"\nAction distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count} ({100*count/len(all_examples):.1f}%)")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
