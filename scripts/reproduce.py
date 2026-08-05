import subprocess
import sys
import os

def run_command(command, description):
    print(f"\n{'='*60}\n🚀 Step: {description}\n{'='*60}")
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"❌ Error during: {description}")
        sys.exit(1)

def main():
    # We will reproduce Table 1 (ZINC) across all 4 variants.
    # Note: Generating all 250 samples for 400 targets takes a very long time. 
    # For reviewer ease, we set n_samples to 25 and samples_per_target to 10 by default.
    # (To perfectly replicate the paper numbers exactly, change to 400 and 250).
    n_targets = "25"
    samples_per_target = "10"

    variants = {
        'base': 'outputs/base_eval',
        'anchor': 'outputs/anchor_eval',
        'sized': 'outputs/sized_eval',
        'both': 'outputs/final_eval' # Matches your compile_results.py CONFIGS
    }

    # Step 1: Generate Molecules for all 4 variants
    for var, out_dir in variants.items():
        os.makedirs(out_dir, exist_ok=True)
        run_command([
            "python", "scripts/generate.py",
            "--checkpoint", "checkpoints/geolinker_best_unified.pt",
            "--fragments", "datasets/zinc/zinc_final_test.pt",
            "--output_dir", out_dir,
            "--variant", var,
            "--n_samples", n_targets, 
            "--samples_per_target", samples_per_target
        ], f"Generating molecules with GeoLinker ({var.upper()} Variant)")

    # Step 2: Use compile_results.py to auto-export, auto-evaluate, and compile the table!
    # Because we added '--run_missing' to compile_results.py, it will detect the 
    # generated .pt files and automatically run export_for_difflinker.py and evaluate.py!
    run_command([
        "python", "scripts/compile_results.py",
        "--run_missing"
    ], "Exporting, Evaluating, and Compiling Final Metrics Table")

    print("\n✅ Reproduction Complete! Scroll up to see the generated LaTeX table.")

if __name__ == "__main__":
    main()