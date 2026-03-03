"""
run_eval.py — CLI Entry Point

Orchestrates the full evaluation pipeline:
    1. Parse config YAML
    2. Run FP32 reference model first, cache reference outputs
    3. Run each quantized variant, compare against FP32 reference
    4. Aggregate metrics across all runs
    5. Output batch summary

Usage:
    python run_eval.py --config config/example_config.yaml --script structured_output
    python run_eval.py --config config/example_config.yaml --script long_context
    python run_eval.py --config config/example_config.yaml --script all
"""

import argparse
import os
import sys
import time
import yaml

from shared.hardware_monitor import HardwareMonitor
from shared.live_logger import LiveLogger
from shared.model_loader import UnifiedModel
from shared.json_builder import JsonBuilder
from shared.metrics_aggregator import MetricsAggregator
from shared.run_tracker import RunTracker

from eval_structured_output import run_structured_output_eval
from eval_long_context import run_long_context_eval


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Evaluation Framework")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument("--script", type=str, required=True,
                        choices=["structured_output", "long_context", "all"],
                        help="Which evaluation script to run")
    return parser.parse_args()


def load_config(config_path):
    """Load and validate YAML config."""
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure output directories exist
    os.makedirs(config.get("output", {}).get("results_dir", "./results"), exist_ok=True)
    os.makedirs(config.get("output", {}).get("logs_dir", "./logs"), exist_ok=True)

    return config


def run_model_evaluation(config, model_group, variant_config, script_type,
                          logger, hw_monitor, json_builder, run_tracker,
                          reference_outputs=None):
    """Run evaluation for a single model variant."""
    model_name = variant_config.get("tag", "unknown")
    quant_type = variant_config.get("quant_type", "unknown")
    bit_depth = variant_config.get("bit_depth", "")

    logger.log("MODEL_LOAD", f"Loading {model_name} ({quant_type} {bit_depth})")

    # Load model
    try:
        model = UnifiedModel(variant_config)
        logger.log("MODEL_READY", model=model_name, quant=quant_type,
                    load_time_s=model.load_time_s)
    except Exception as e:
        logger.log("ERROR", f"Failed to load model {model_name}: {e}")
        return []

    # Set model-level overrides in config for result building
    config["_current_model_family"] = model_group.get("family", "")
    config["_current_model_doc_sizes"] = model_group.get("long_context_doc_sizes")
    config["_current_model_max_context"] = model_group.get("max_context_tokens")

    results = []

    # Run evaluation(s)
    if script_type in ("structured_output", "all"):
        so_results = run_structured_output_eval(
            config, model, logger, hw_monitor, json_builder, run_tracker,
            reference_outputs=reference_outputs.get("structured_output") if reference_outputs else None
        )
        results.extend(so_results)

    if script_type in ("long_context", "all"):
        lc_results = run_long_context_eval(
            config, model, logger, hw_monitor, json_builder, run_tracker,
            reference_outputs=reference_outputs.get("long_context") if reference_outputs else None
        )
        results.extend(lc_results)

    return results


def extract_reference_outputs(results):
    """Extract reference outputs from FP32 results for comparison."""
    refs = {
        "structured_output": {},
        "long_context": {},
    }

    for result in results:
        meta = result.get("test_meta", {})
        script_type = meta.get("script_type", "")
        output_data = result.get("output", {})

        if script_type == "structured_output":
            level = result.get("task_specific", {}).get("schema_complexity_level")
            if level and output_data.get("parsed_json"):
                refs["structured_output"][level] = output_data["parsed_json"]

        elif script_type == "long_context":
            doc_size = result.get("task_specific", {}).get("doc_size")
            depth = result.get("task_specific", {}).get("needle_depth_percent")
            if doc_size and depth:
                key = f"{doc_size}_needle{depth}"
                refs["long_context"][key] = output_data

    return refs


def main():
    args = parse_args()
    config = load_config(args.config)

    results_dir = config.get("output", {}).get("results_dir", "./results")
    logs_dir = config.get("output", {}).get("logs_dir", "./logs")
    cost_per_hour = config.get("hardware", {}).get("cost_per_hour_usd")

    # Initialize shared components
    hw_monitor = HardwareMonitor()
    json_builder = JsonBuilder(results_dir, cost_per_hour_usd=cost_per_hour)
    run_tracker = RunTracker(results_dir)
    aggregator = MetricsAggregator()

    # Try to rebuild manifest from existing results
    run_tracker.rebuild_from_results()

    # Print system info
    sys_info = hw_monitor.get_system_info()
    print("\n" + "=" * 60)
    print("  LLM Evaluation Framework")
    print("=" * 60)
    print(f"  Script:   {args.script}")
    print(f"  Config:   {args.config}")
    print(f"  GPU:      {sys_info.get('gpu_model', 'Not detected')}")
    print(f"  VRAM:     {sys_info.get('gpu_vram_total_mb', 'N/A')} MB")
    print(f"  CUDA:     {sys_info.get('cuda_version', 'N/A')}")
    print(f"  Node ID:  {sys_info.get('salad_node_id', 'local')}")
    print("=" * 60 + "\n")

    all_results = []
    models = config.get("models", [])

    for model_group in models:
        model_name = model_group.get("name", "Unknown")
        variants = model_group.get("variants", [])

        print(f"\n{'─' * 50}")
        print(f"  Model: {model_name}")
        print(f"  Variants: {len(variants)}")
        print(f"{'─' * 50}\n")

        # Find reference variant (FP32)
        ref_variant = next((v for v in variants if v.get("is_reference")), None)
        reference_outputs = None

        # --- Step 1: Run FP32 reference FIRST ---
        if ref_variant:
            logger = LiveLogger(
                f"{model_name}_FP32_reference",
                logs_dir
            )

            print(f"  [1/2] Running FP32 reference: {ref_variant['tag']}")
            ref_results = run_model_evaluation(
                config, model_group, ref_variant, args.script,
                logger, hw_monitor, json_builder, run_tracker
            )
            all_results.extend(ref_results)
            reference_outputs = extract_reference_outputs(ref_results)
            logger.close()

        # --- Step 2: Run quantized variants ---
        non_ref_variants = [v for v in variants if not v.get("is_reference")]

        for i, variant in enumerate(non_ref_variants, 1):
            quant_label = variant.get("quant_type", "")
            bit_depth = variant.get("bit_depth", "")
            label = f"{quant_label}_{bit_depth}" if bit_depth else quant_label

            logger = LiveLogger(
                f"{model_name}_{label}",
                logs_dir
            )

            print(f"  [{i + 1}/{len(variants)}] Running {label}: {variant['tag']}")
            variant_results = run_model_evaluation(
                config, model_group, variant, args.script,
                logger, hw_monitor, json_builder, run_tracker,
                reference_outputs=reference_outputs
            )
            all_results.extend(variant_results)
            logger.close()

    # --- Step 3: Aggregate and output batch summary ---
    print(f"\n{'=' * 60}")
    print(f"  Aggregating results...")
    print(f"{'=' * 60}\n")

    summary = aggregator.aggregate(all_results)
    aggregator.save_and_output(summary, results_dir)

    # Final stats
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION COMPLETE")
    print(f"  Total runs:     {summary.get('total_runs', 0)}")
    print(f"  Successful:     {summary.get('successful_runs', 0)}")
    print(f"  Failed:         {summary.get('failed_runs', 0)}")
    print(f"  p50 latency:    {summary.get('p50_latency_ms', 'N/A')} ms")
    print(f"  p95 latency:    {summary.get('p95_latency_ms', 'N/A')} ms")
    print(f"  Error rate:     {summary.get('error_rate_percent', 0)}%")
    print(f"  Results dir:    {results_dir}")
    print(f"{'=' * 60}\n")

    # Cleanup
    hw_monitor.cleanup()


if __name__ == "__main__":
    main()
