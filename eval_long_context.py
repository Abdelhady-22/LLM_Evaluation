"""
eval_long_context.py — Script 2: Long Context Evaluator

Flow:
    1. Load model via unified loader
    2. Load base document at configured size (4K/8K/16K/32K)
    3. Inject needle fact at configured depth (25/50/75/90%)
    4. Build 3-part input: system prompt + context document + user question
    5. Generate response with performance monitoring
    6. Compare against FP32 reference output for same task
    7. Evaluate: needle found?, answer relevance, faithfulness, instruction following
    8. Compute degradation score vs reference
    9. Save full JSON result
"""


class LongContextEvaluator:
    """Evaluates model ability to process and reason over long contexts."""

    def __init__(self, config, logger, hardware_monitor, json_builder):
        # TODO: Implement
        pass

    def run(self, model, doc_size, needle_depth=None, reference_output=None):
        # TODO: Implement — full evaluation pipeline for one task
        raise NotImplementedError

    def _load_document(self, doc_size):
        # TODO: Implement — load base document by size
        raise NotImplementedError

    def _inject_needle(self, document, needle, depth_percent):
        # TODO: Implement — insert needle fact at specified depth
        raise NotImplementedError

    def _build_prompt(self, system_prompt, context, user_prompt):
        # TODO: Implement — assemble 3-part input
        raise NotImplementedError

    def _evaluate_response(self, response, reference_output, needle=None):
        # TODO: Implement — score relevance, faithfulness, needle detection
        raise NotImplementedError

    def _compare_with_reference(self, test_output, reference_output):
        # TODO: Implement — compute similarity vs FP32 reference
        raise NotImplementedError


def run_long_context_eval(config):
    """Entry point for long context evaluation."""
    # TODO: Implement
    pass
