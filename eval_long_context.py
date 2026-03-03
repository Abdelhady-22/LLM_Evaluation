"""
eval_long_context.py — Script 2: Long Context Evaluator

Evaluates model ability to process long documents and find injected facts.
Compares quantized model output against FP32 reference baseline.
"""

import json
import os
import time
import difflib

from shared.model_loader import UnifiedModel
from shared.live_logger import LiveLogger
from shared.hardware_monitor import HardwareMonitor
from shared.json_builder import JsonBuilder
from shared.run_tracker import RunTracker


# File paths
DOCS_DIR = os.path.join(os.path.dirname(__file__), "tasks", "long_context", "documents")
NEEDLES_PATH = os.path.join(os.path.dirname(__file__), "tasks", "long_context", "needles", "needle_facts.json")
TASKS_PATH = os.path.join(os.path.dirname(__file__), "tasks", "long_context", "prompts", "context_tasks.json")

DOC_SIZE_MAP = {
    "4k": "doc_4k.txt",
    "7k": "doc_7k.txt",
    "8k": "doc_8k.txt",
    "16k": "doc_16k.txt",
    "32k": "doc_32k.txt",
    "64k": "doc_64k.txt",
    "120k": "doc_120k.txt",
}


class LongContextEvaluator:
    """Evaluates model ability to process and reason over long contexts."""

    def __init__(self, config, logger, hardware_monitor, json_builder):
        self.config = config
        self.logger = logger
        self.hw_monitor = hardware_monitor
        self.json_builder = json_builder
        self.gen_params = config.get("generation_defaults", {})

        # Load needle facts and task definitions
        self.needles = self._load_needles()
        self.context_tasks = self._load_context_tasks()

    def run(self, model, doc_size, needle_depth, reference_output=None):
        """Run full evaluation for one model + doc size + needle depth."""
        self.logger.log("TASK_START", f"long_context {doc_size} needle@{needle_depth}%",
                        model=model.model_name, doc_size=doc_size, needle_depth=needle_depth)

        # Load document
        document = self._load_document(doc_size)
        if not document or document.startswith("TODO:"):
            self.logger.log("ERROR", f"Document {doc_size} not ready (placeholder)")
            return None

        # Select needle and inject at depth
        needle = self.needles[0]  # Use first needle by default
        injected_doc = self._inject_needle(document, needle["fact"], needle_depth)

        # Get the needle task definition
        needle_task = next((t for t in self.context_tasks if t["id"] == "needle_task"), self.context_tasks[0])

        # Build messages
        system_prompt = needle_task["system_prompt"]
        user_prompt = needle["question"]
        messages = self._build_prompt(system_prompt, injected_doc, user_prompt)

        # Track VRAM
        vram_before = self.hw_monitor.get_vram_used_mb()
        self.hw_monitor.start_monitoring()

        # Generate
        self.logger.log("GENERATING", model=model.model_name,
                        doc_size=doc_size, needle_depth=needle_depth)

        gen_result = model.generate(messages, self.gen_params)

        # Stop monitoring
        self.hw_monitor.stop_monitoring()
        hw_report = self.hw_monitor.get_report()
        vram_after = self.hw_monitor.get_vram_used_mb()

        if gen_result.get("error"):
            self.logger.log("ERROR", gen_result["error"], model=model.model_name)
        else:
            if gen_result.get("ttft_ms"):
                self.logger.log("FIRST_TOKEN", ttft_ms=gen_result["ttft_ms"])
            self.logger.log("COMPLETE",
                            latency_ms=gen_result.get("total_latency_ms"),
                            output_tokens=gen_result.get("output_tokens"),
                            tps=round(gen_result["output_tokens"] / (gen_result["total_latency_ms"] / 1000), 1)
                            if gen_result.get("total_latency_ms", 0) > 0 else 0)

        # Evaluate response
        response_text = gen_result.get("response_text", "")
        evaluation = self._evaluate_response(response_text, needle, reference_output)

        # Score
        quality_scores = self._score(evaluation)
        self.logger.log("SCORING", f"needle_found={evaluation.get('needle_found')}",
                        similarity=evaluation.get("value_similarity_score"))

        model_info = model.get_model_info()

        # Build result
        result = self.json_builder.build_result(
            test_meta={
                "script_type": "long_context",
                "task_name": f"long_context_{doc_size}_needle{needle_depth}",
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
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "doc_size": doc_size,
                "doc_token_count_approx": len(document.split()),
                "needle_fact": needle["fact"],
                "needle_depth_percent": needle_depth,
                "input_tokens": gen_result.get("input_tokens", 0),
            },
            output_data={
                "raw_response": response_text,
                "output_tokens": gen_result.get("output_tokens", 0),
            },
            quality_scores=quality_scores,
            performance={
                "model_load_time_s": model.load_time_s,
                "ttft_ms": gen_result.get("ttft_ms"),
                "total_latency_ms": gen_result.get("total_latency_ms"),
                "tokens_per_second": round(
                    gen_result["output_tokens"] / (gen_result["total_latency_ms"] / 1000), 1
                ) if gen_result.get("total_latency_ms", 0) > 0 else None,
                "input_tokens": gen_result.get("input_tokens", 0),
                "output_tokens": gen_result.get("output_tokens", 0),
            },
            logs={
                "log_file_path": self.logger.log_file_path,
                "key_events": self.logger.get_key_events()[-10:],
                "error_counts": self.logger.get_error_counts(),
            },
            task_specific={
                "doc_size": doc_size,
                "needle_depth_percent": needle_depth,
                "needle_id": needle["id"],
                "needle_question": needle["question"],
            },
            verdict={
                "passed": evaluation.get("needle_found", False) and not gen_result.get("error"),
                "failure_reason": gen_result.get("error") if gen_result.get("error") else
                                  ("needle not found" if not evaluation.get("needle_found") else None),
                "notes": "",
            },
        )

        filepath = self.json_builder.save_and_output(result)
        self.logger.log("SAVED", filepath)

        return result

    def _load_document(self, doc_size):
        """Load base document by size."""
        filename = DOC_SIZE_MAP.get(doc_size)
        if not filename:
            return None
        filepath = os.path.join(DOCS_DIR, filename)
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def _load_needles(self):
        """Load needle facts from JSON file."""
        if not os.path.exists(NEEDLES_PATH):
            return [{"id": "default", "fact": "The secret code is 42.", "question": "What is the secret code?"}]
        with open(NEEDLES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("needles", [])

    def _load_context_tasks(self):
        """Load context task definitions."""
        if not os.path.exists(TASKS_PATH):
            return [{"id": "default", "system_prompt": "Answer based on the document.", "user_prompt": "Summarize."}]
        with open(TASKS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tasks", [])

    def _inject_needle(self, document, needle_fact, depth_percent):
        """Insert needle fact at specified depth percentage in the document."""
        lines = document.split("\n")
        if not lines:
            return needle_fact

        # Calculate insertion position
        insert_idx = max(0, int(len(lines) * (depth_percent / 100.0)))
        insert_idx = min(insert_idx, len(lines))

        # Insert needle with surrounding blank lines for natural feel
        needle_line = f"\n{needle_fact}\n"
        lines.insert(insert_idx, needle_line)

        return "\n".join(lines)

    def _build_prompt(self, system_prompt, context, user_prompt):
        """Assemble 3-part input: system + context + question."""
        user_content = (
            f"DOCUMENT:\n"
            f"---\n"
            f"{context}\n"
            f"---\n\n"
            f"QUESTION: {user_prompt}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _evaluate_response(self, response, needle, reference_output=None):
        """Evaluate response: needle found?, relevance, similarity to reference."""
        evaluation = {
            "needle_found": False,
            "answer_relevance": 0.0,
            "faithfulness": 0.0,
        }

        if not response:
            return evaluation

        response_lower = response.lower()

        # Check needle detection — look for key facts from the needle
        needle_fact = needle["fact"].lower()
        # Extract key terms from the needle fact (numbers, names, specific terms)
        key_terms = self._extract_key_terms(needle_fact)
        terms_found = sum(1 for term in key_terms if term in response_lower)
        needle_score = terms_found / len(key_terms) if key_terms else 0

        evaluation["needle_found"] = needle_score >= 0.5  # At least half the key terms found
        evaluation["needle_detection_score"] = round(needle_score, 3)

        # Answer relevance — does the response address the question?
        question_terms = self._extract_key_terms(needle["question"].lower())
        relevance_found = sum(1 for term in question_terms if term in response_lower)
        evaluation["answer_relevance"] = round(
            relevance_found / len(question_terms) if question_terms else 0, 3
        )

        # Faithfulness — is the response based on document content (not hallucinated)?
        # Simple heuristic: check if response doesn't contain "I don't know" or "not found"
        negative_phrases = ["i don't know", "not found", "not mentioned", "no information",
                           "cannot determine", "unable to find"]
        has_negative = any(phrase in response_lower for phrase in negative_phrases)
        evaluation["faithfulness"] = 0.2 if has_negative else 0.8

        # Compare with reference if available
        if reference_output:
            ref_text = reference_output.get("raw_response", "")
            if ref_text:
                similarity = difflib.SequenceMatcher(None, response.lower(),
                                                       ref_text.lower()).ratio()
                evaluation["value_similarity_score"] = round(similarity, 3)
                evaluation["reference_match_rate"] = round(similarity, 3)

        return evaluation

    def _extract_key_terms(self, text):
        """Extract key terms from text (numbers, capitalized words, specific nouns)."""
        import re
        terms = []

        # Extract numbers (including decimals and currency)
        numbers = re.findall(r'\$?[\d,]+\.?\d*', text)
        terms.extend([n.lower().replace(",", "") for n in numbers])

        # Extract multi-word proper nouns and specific terms
        words = text.split()
        for word in words:
            cleaned = word.strip(".,;:!?\"'()[]{}").lower()
            # Keep meaningful words (longer than 3 chars, not common words)
            common = {"the", "and", "for", "was", "that", "with", "this", "from",
                      "are", "has", "have", "had", "been", "were", "its", "into",
                      "what", "when", "where", "which", "who", "how", "she", "her"}
            if len(cleaned) > 3 and cleaned not in common:
                terms.append(cleaned)

        return list(set(terms))

    def _score(self, evaluation):
        """Compute final quality scores dict."""
        scores = {
            "needle_found": evaluation.get("needle_found", False),
            "needle_detection_score": evaluation.get("needle_detection_score", 0),
            "answer_relevance": evaluation.get("answer_relevance", 0),
            "faithfulness": evaluation.get("faithfulness", 0),
        }

        if "reference_match_rate" in evaluation:
            scores["reference_match_rate"] = evaluation["reference_match_rate"]
            scores["value_similarity_score"] = evaluation["value_similarity_score"]

        # Overall score
        overall = (
            (1.0 if scores["needle_found"] else 0.0) * 0.35 +
            scores["needle_detection_score"] * 0.15 +
            scores["answer_relevance"] * 0.25 +
            scores["faithfulness"] * 0.25
        )
        scores["overall_score"] = round(overall, 3)

        return scores


def run_long_context_eval(config, model, logger, hw_monitor, json_builder, run_tracker,
                           reference_outputs=None):
    """Entry point: run all doc sizes × needle depths for one model."""
    evaluator = LongContextEvaluator(config, logger, hw_monitor, json_builder)

    # Use per-model doc sizes if set, otherwise fall back to global config
    doc_sizes = config.get("_current_model_doc_sizes") or \
                config.get("long_context", {}).get("doc_sizes", ["4k", "8k", "16k", "32k"])
    needle_depths = config.get("long_context", {}).get("needle_depths", [25, 50, 75, 90])

    max_ctx = config.get("_current_model_max_context")
    if max_ctx:
        logger.log("CONFIG", f"Model context window: {max_ctx} tokens, doc sizes: {doc_sizes}")
    results = []

    for doc_size in doc_sizes:
        for depth in needle_depths:
            run_key = RunTracker.make_run_key(
                model.model_name, "long_context",
                f"long_context_{doc_size}_needle{depth}",
                doc_size=doc_size, needle_depth=depth
            )

            if run_tracker.is_completed(run_key):
                logger.log("TASK_START",
                           f"SKIPPED (already completed) long_context {doc_size} needle@{depth}%",
                           model=model.model_name)
                continue

            ref_output = None
            if reference_outputs:
                ref_key = f"{doc_size}_needle{depth}"
                ref_output = reference_outputs.get(ref_key)

            try:
                result = evaluator.run(model, doc_size, depth, reference_output=ref_output)
                if result:
                    results.append(result)
                    run_tracker.mark_completed(run_key, "")
                else:
                    logger.log("ERROR", f"Skipped {doc_size} (document not ready)")
            except Exception as e:
                logger.log("ERROR", f"Failed: {e}", model=model.model_name,
                           doc_size=doc_size, needle_depth=depth)
                run_tracker.mark_failed(run_key, str(e))

    return results
