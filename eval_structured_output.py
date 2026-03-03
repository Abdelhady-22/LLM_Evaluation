"""
eval_structured_output.py — Script 1: Structured Output Evaluator

Evaluates model ability to produce valid, schema-compliant JSON output.
Compares quantized model output against FP32 reference baseline.
"""

import json
import os
import time
import difflib

import jsonschema

from shared.model_loader import UnifiedModel
from shared.live_logger import LiveLogger
from shared.hardware_monitor import HardwareMonitor
from shared.json_builder import JsonBuilder
from shared.run_tracker import RunTracker


# Schema and prompt file paths (relative to project root)
SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "tasks", "structured_output", "schemas")
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "tasks", "structured_output", "prompts")

LEVEL_CONFIG = {
    "L1": {"schema": "l1_flat.json", "prompt": "l1_task.txt"},
    "L3": {"schema": "l3_nested.json", "prompt": "l3_task.txt"},
    "L5": {"schema": "l5_complex.json", "prompt": "l5_task.txt"},
}


class StructuredOutputEvaluator:
    """Evaluates model ability to produce valid structured JSON output."""

    def __init__(self, config, logger, hardware_monitor, json_builder):
        self.config = config
        self.logger = logger
        self.hw_monitor = hardware_monitor
        self.json_builder = json_builder
        self.max_retries = config.get("structured_output", {}).get("max_retries", 1)
        self.gen_params = config.get("generation_defaults", {})

    def run(self, model, level, reference_output=None):
        """Run full evaluation pipeline for one model + one complexity level."""
        self.logger.log("TASK_START", f"structured_output {level}",
                        model=model.model_name, level=level)

        # Load schema and prompt
        schema = self._load_schema(level)
        task_prompt = self._load_prompt(level)

        # Build messages
        messages = self._build_prompt(schema, task_prompt)

        # Track VRAM before generation
        vram_before = self.hw_monitor.get_vram_used_mb()

        # Start hardware monitoring for this task
        self.hw_monitor.start_monitoring()

        # Generate response (with retry logic)
        attempts = 0
        total_tokens_wasted = 0
        last_error = None
        gen_result = None
        parse_success = False
        parsed_json = None

        while attempts <= self.max_retries:
            attempts += 1
            self.logger.log("GENERATING", f"attempt {attempts}/{self.max_retries + 1}",
                            model=model.model_name)

            gen_result = model.generate(messages, self.gen_params)

            if gen_result.get("error"):
                self.logger.log("ERROR", gen_result["error"],
                                model=model.model_name, attempt=attempts)
                last_error = gen_result["error"]
                continue

            if gen_result.get("ttft_ms"):
                self.logger.log("FIRST_TOKEN", ttft_ms=gen_result["ttft_ms"])

            # Try to parse JSON
            parsed_json, parse_error = self._try_parse_json(gen_result["response_text"])

            if parsed_json is not None:
                parse_success = True
                self.logger.log("COMPLETE",
                                latency_ms=gen_result["total_latency_ms"],
                                output_tokens=gen_result["output_tokens"],
                                tps=round(gen_result["output_tokens"] / (gen_result["total_latency_ms"] / 1000), 1)
                                if gen_result["total_latency_ms"] > 0 else 0)
                break
            else:
                # Parse failed — inject error into retry prompt
                total_tokens_wasted += gen_result.get("output_tokens", 0)
                last_error = parse_error
                self.logger.log("ERROR", f"JSON parse failed: {parse_error}",
                                attempt=attempts)

                if attempts <= self.max_retries:
                    messages = self._build_retry_prompt(messages, gen_result["response_text"], parse_error)

        # Stop monitoring
        self.hw_monitor.stop_monitoring()
        hw_report = self.hw_monitor.get_report()
        vram_after = self.hw_monitor.get_vram_used_mb()

        # Validate against schema
        validation_result = self._validate_output(parsed_json, schema) if parsed_json else {
            "schema_compliance": 0,
            "field_completeness": 0,
            "type_correctness": 0,
            "hallucinated_fields": 0,
            "errors": [str(last_error)],
        }

        # Compare with reference
        reference_comparison = {}
        if reference_output and parsed_json:
            reference_comparison = self._compare_with_reference(parsed_json, reference_output)

        # Score
        quality_scores = self._score(parse_success, validation_result, reference_comparison)

        # Build result
        self.logger.log("SCORING", "computing quality scores",
                        parse_success=parse_success,
                        schema_compliance=quality_scores.get("schema_compliance"))

        model_info = model.get_model_info()

        result = self.json_builder.build_result(
            test_meta={
                "script_type": "structured_output",
                "task_name": f"structured_output_{level}",
                "complexity_level": level,
            },
            model_config={
                "family": self.config.get("_current_model_family", ""),
                **model_info,
            },
            hardware={
                "vram_before_task_mb": vram_before,
                "vram_after_task_mb": vram_after,
                **hw_report,
            },
            input_data={
                "system_prompt": messages[0]["content"] if messages else "",
                "user_prompt": messages[-1]["content"] if messages else "",
                "target_schema": schema,
                "input_tokens": gen_result.get("input_tokens", 0) if gen_result else 0,
            },
            output_data={
                "raw_response": gen_result.get("response_text", "") if gen_result else "",
                "parsed_json": parsed_json,
                "parse_success": parse_success,
                "attempts_needed": attempts,
                "output_tokens": gen_result.get("output_tokens", 0) if gen_result else 0,
            },
            quality_scores=quality_scores,
            performance={
                "model_load_time_s": model.load_time_s,
                "ttft_ms": gen_result.get("ttft_ms") if gen_result else None,
                "total_latency_ms": gen_result.get("total_latency_ms") if gen_result else None,
                "tokens_per_second": round(
                    gen_result["output_tokens"] / (gen_result["total_latency_ms"] / 1000), 1
                ) if gen_result and gen_result.get("total_latency_ms", 0) > 0 else None,
                "input_tokens": gen_result.get("input_tokens", 0) if gen_result else 0,
                "output_tokens": gen_result.get("output_tokens", 0) if gen_result else 0,
                "retry_count": attempts - 1,
                "tokens_wasted_on_retries": total_tokens_wasted,
            },
            logs={
                "log_file_path": self.logger.log_file_path,
                "key_events": self.logger.get_key_events()[-10:],
                "error_counts": self.logger.get_error_counts(),
            },
            task_specific={
                "schema_complexity_level": level,
                "schema_field_count": self._count_schema_fields(schema),
            },
            verdict={
                "passed": parse_success and quality_scores.get("schema_compliance", 0) > 0.5,
                "failure_reason": str(last_error) if not parse_success else None,
                "notes": "",
            },
        )

        # Save and output
        filepath = self.json_builder.save_and_output(result)
        self.logger.log("SAVED", filepath)

        return result

    def _load_schema(self, level):
        """Load JSON schema for the given complexity level."""
        schema_file = os.path.join(SCHEMA_DIR, LEVEL_CONFIG[level]["schema"])
        with open(schema_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_prompt(self, level):
        """Load task prompt for the given complexity level."""
        prompt_file = os.path.join(PROMPT_DIR, LEVEL_CONFIG[level]["prompt"])
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _build_prompt(self, schema, task_prompt):
        """Build messages with system prompt (schema) + user prompt (task)."""
        schema_str = json.dumps(schema, indent=2)

        system_prompt = (
            "You are a data extraction assistant. Your task is to analyze the given content "
            "and return ONLY a valid JSON object that matches the following JSON schema exactly.\n\n"
            "RULES:\n"
            "1. Return ONLY valid JSON — no markdown, no explanation, no extra text.\n"
            "2. All required fields MUST be present.\n"
            "3. Field types MUST match the schema (string, number, array, etc.).\n"
            "4. Do NOT add fields that are not in the schema.\n"
            "5. Use meaningful values based on the input — do not use placeholder text.\n\n"
            f"TARGET SCHEMA:\n```json\n{schema_str}\n```"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

    def _build_retry_prompt(self, original_messages, failed_response, error):
        """Build retry prompt with previous error injected."""
        retry_message = (
            f"Your previous response was not valid JSON. Error: {error}\n\n"
            f"Your previous output was:\n{failed_response[:500]}\n\n"
            "Please try again. Return ONLY valid JSON matching the schema. "
            "No explanation, no markdown code blocks, just the raw JSON object."
        )

        return original_messages + [
            {"role": "assistant", "content": failed_response},
            {"role": "user", "content": retry_message},
        ]

    def _try_parse_json(self, response_text):
        """Try to parse JSON from response text. Returns (parsed_json, error)."""
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            parsed = json.loads(text)
            return parsed, None
        except json.JSONDecodeError as e:
            return None, str(e)

    def _validate_output(self, parsed_json, schema):
        """Validate parsed JSON against schema. Returns validation metrics."""
        errors = []
        try:
            jsonschema.validate(instance=parsed_json, schema=schema)
        except jsonschema.ValidationError as e:
            errors.append(str(e.message))
        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e.message}")

        # Field completeness
        required_fields = schema.get("required", [])
        present_fields = set(parsed_json.keys()) if isinstance(parsed_json, dict) else set()
        required_set = set(required_fields)
        field_completeness = len(present_fields & required_set) / len(required_set) if required_set else 1.0

        # Hallucinated fields
        schema_fields = set(schema.get("properties", {}).keys())
        hallucinated = present_fields - schema_fields
        hallucinated_count = len(hallucinated)

        # Type correctness (check each field)
        type_correct = 0
        type_total = 0
        for field, field_schema in schema.get("properties", {}).items():
            if field in parsed_json:
                type_total += 1
                expected_type = field_schema.get("type")
                value = parsed_json[field]
                if self._check_type(value, expected_type):
                    type_correct += 1

        type_correctness = type_correct / type_total if type_total > 0 else 1.0

        # Schema compliance (overall)
        schema_compliance = 1.0 if not errors else max(0, 1.0 - (len(errors) * 0.2))

        return {
            "schema_compliance": round(schema_compliance, 2),
            "field_completeness": round(field_completeness, 2),
            "type_correctness": round(type_correctness, 2),
            "hallucinated_fields": hallucinated_count,
            "errors": errors,
        }

    def _check_type(self, value, expected_type):
        """Check if a value matches the expected JSON schema type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True
        return isinstance(value, expected_python_type)

    def _compare_with_reference(self, test_output, reference_output):
        """Compare test output against FP32 reference. Field-by-field match + similarity."""
        if not isinstance(test_output, dict) or not isinstance(reference_output, dict):
            return {"reference_match_rate": 0, "value_similarity_score": 0}

        all_keys = set(test_output.keys()) | set(reference_output.keys())
        exact_matches = 0
        similarity_scores = []

        for key in all_keys:
            test_val = test_output.get(key)
            ref_val = reference_output.get(key)

            if test_val == ref_val:
                exact_matches += 1
                similarity_scores.append(1.0)
            elif test_val is not None and ref_val is not None:
                # Fuzzy compare for strings
                if isinstance(test_val, str) and isinstance(ref_val, str):
                    ratio = difflib.SequenceMatcher(None, test_val, ref_val).ratio()
                    similarity_scores.append(ratio)
                elif isinstance(test_val, (int, float)) and isinstance(ref_val, (int, float)):
                    # Numeric closeness
                    if ref_val != 0:
                        closeness = 1.0 - min(abs(test_val - ref_val) / abs(ref_val), 1.0)
                    else:
                        closeness = 1.0 if test_val == 0 else 0.0
                    similarity_scores.append(closeness)
                else:
                    # Compare JSON representations
                    test_str = json.dumps(test_val, sort_keys=True)
                    ref_str = json.dumps(ref_val, sort_keys=True)
                    ratio = difflib.SequenceMatcher(None, test_str, ref_str).ratio()
                    similarity_scores.append(ratio)
            else:
                similarity_scores.append(0.0)

        match_rate = exact_matches / len(all_keys) if all_keys else 1.0
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

        return {
            "reference_match_rate": round(match_rate, 3),
            "value_similarity_score": round(avg_similarity, 3),
            "total_fields_compared": len(all_keys),
            "exact_matches": exact_matches,
        }

    def _score(self, parse_success, validation_result, reference_comparison):
        """Compute final quality_scores dict."""
        scores = {
            "parse_success": parse_success,
            "schema_compliance": validation_result.get("schema_compliance", 0),
            "field_completeness": validation_result.get("field_completeness", 0),
            "type_correctness": validation_result.get("type_correctness", 0),
            "hallucinated_fields": validation_result.get("hallucinated_fields", 0),
        }

        if reference_comparison:
            scores["reference_match_rate"] = reference_comparison.get("reference_match_rate", 0)
            scores["value_similarity_score"] = reference_comparison.get("value_similarity_score", 0)

        # Overall score (weighted average)
        if parse_success:
            overall = (
                scores["schema_compliance"] * 0.3 +
                scores["field_completeness"] * 0.25 +
                scores["type_correctness"] * 0.2 +
                scores.get("value_similarity_score", scores.get("reference_match_rate", 0.5)) * 0.25
            )
        else:
            overall = 0.0

        scores["overall_score"] = round(overall, 3)
        return scores

    def _count_schema_fields(self, schema):
        """Count total number of fields in a schema (recursive)."""
        count = 0
        props = schema.get("properties", {})
        for field_name, field_def in props.items():
            count += 1
            if field_def.get("type") == "object":
                count += self._count_schema_fields(field_def)
            elif field_def.get("type") == "array" and "items" in field_def:
                items = field_def["items"]
                if items.get("type") == "object":
                    count += self._count_schema_fields(items)
        return count


def run_structured_output_eval(config, model, logger, hw_monitor, json_builder, run_tracker,
                                reference_outputs=None):
    """Entry point: run all structured output levels for one model."""
    evaluator = StructuredOutputEvaluator(config, logger, hw_monitor, json_builder)
    levels = config.get("structured_output", {}).get("levels", ["L1", "L3", "L5"])
    results = []

    for level in levels:
        run_key = RunTracker.make_run_key(model.model_name, "structured_output",
                                           f"structured_output_{level}", level=level)

        if run_tracker.is_completed(run_key):
            logger.log("TASK_START", f"SKIPPED (already completed) structured_output {level}",
                        model=model.model_name)
            continue

        ref_output = None
        if reference_outputs and level in reference_outputs:
            ref_output = reference_outputs[level]

        try:
            result = evaluator.run(model, level, reference_output=ref_output)
            results.append(result)
            run_tracker.mark_completed(run_key, result.get("logs", {}).get("log_file_path", ""))
        except Exception as e:
            logger.log("ERROR", f"Failed: {e}", model=model.model_name, level=level)
            run_tracker.mark_failed(run_key, str(e))

    return results
