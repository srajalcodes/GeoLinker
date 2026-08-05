"""
geoLinker v8 — compile_results.py
Automates the evaluation and compilation of the entire comparative metrics table
across ZINC, CASF, and GEOM datasets for all four GeoLinker variants.
"""

import os
import argparse
import subprocess
import pandas as pd
import numpy as np

# Map datasets and variants to their respective folders and data paths
CONFIGS = {
    'ZINC': {
        'base': ('outputs/base_eval', 'datasets/zinc/zinc_final_test.pt', 'datasets/zinc/zinc_final_train_linkers.smi'),
        'anchors': ('outputs/anchor_eval', 'datasets/zinc/zinc_final_test.pt', 'datasets/zinc/zinc_final_train_linkers.smi'),
        'sized': ('outputs/sized_eval', 'datasets/zinc/zinc_final_test.pt', 'datasets/zinc/zinc_final_train_linkers.smi'),
        'both': ('outputs/final_eval', 'datasets/zinc/zinc_final_test.pt', 'datasets/zinc/zinc_final_train_linkers.smi'),
    },
    'CASF': {
        'base': ('outputs/casf_base_eval', 'datasets/casf/casf_final_test.pt', 'datasets/zinc/zinc_final_train_linkers.smi'),
        'anchors': ('outputs/casf_anchor_eval', 'datasets/casf/casf_final_test.pt', 'datasets/zinc/zinc_final_train_linkers.smi'),
        'sized': ('outputs/casf_sized_eval', 'datasets/casf/casf_final_test.pt', 'datasets/zinc/zinc_final_train_linkers.smi'),
        'both': ('outputs/casf_both_eval', 'datasets/casf/casf_final_test.pt', 'datasets/zinc/zinc_final_train_linkers.smi'),
    },
    'GEOM': {
        'base': ('outputs/geom_base_eval', 'datasets/geom/geom_multifrag_test.pt', 'datasets/zinc/zinc_final_train_linkers.smi'),
        'anchors': ('outputs/geom_anchor_eval', 'datasets/geom/geom_multifrag_test.pt', 'datasets/zinc/zinc_final_train_linkers.smi'),
        'sized': ('outputs/geom_sized_eval', 'datasets/geom/geom_multifrag_test.pt', 'datasets/zinc/zinc_final_train_linkers.smi'),
        'both': ('outputs/geom_both_eval', 'datasets/geom/geom_multifrag_test.pt', 'datasets/zinc/zinc_final_train_linkers.smi'),
    }
}

# Baseline rows hardcoded to preserve your exact LaTeX formatting
BASELINES = {
    'ZINC': r"""DeLinker+ConfVAE+MMFF$^\dagger$ & 0.64 & 3.11 & 0.21 & 98.3 & 44.2 & 47.1 & 33.8 & 0.56 \\
3DLinker$^\dagger$ & 0.65 & 3.14 & 0.24 & 71.5 & 29.2 & 41.9 & 28.0 & 0.52 \\
3DLinker (given anchors)$^\dagger$ & 0.65 & 3.11 & 0.23 & 99.3 & 29.0 & 41.2 & 33.8 & 0.52 \\
DiffLinker$^\dagger$ & 0.68 & 3.01 & 0.25 & 93.8 & 24.0 & 30.3 & 74.0 & 0.72 \\
DiffLinker (given anchors)$^\dagger$ & 0.68 & 3.03 & 0.26 & 97.6 & 22.7 & 32.4 & 74.0 & 0.72 \\
DiffLinker (sampled size)$^\dagger$ & 0.65 & 3.19 & 0.32 & 90.6 & 51.4 & 42.9 & 71.0 & 0.68 \\
DiffLinker (anchors+size)$^\dagger$ & 0.65 & 3.24 & 0.36 & 94.8 & 50.9 & 47.7 & 71.0 & 0.68 \\""",
    'CASF': r"""DiffLinker$^\dagger$ & 0.41 & 4.00 & 0.34 & 85.3 & 40.5 & 41.8 & 36.5 & 0.75 \\
DiffLinker (given anchors)$^\dagger$ & 0.40 & 4.03 & 0.38 & 90.2 & 37.3 & 48.4 & 37.8 & 0.76 \\
DiffLinker (sampled size)$^\dagger$ & 0.40 & 4.06 & 0.30 & 63.7 & 60.0 & 49.3 & 31.2 & 0.75 \\
DiffLinker (anchors+size)$^\dagger$ & 0.40 & 4.10 & 0.38 & 68.4 & 57.1 & 56.9 & 31.1 & 0.72 \\""",
    'GEOM': r"""DiffLinker$^\dagger$ & 0.48 & 2.99 & 0.75 & 93.4 & 31.7 & 68.6 & 89.1 & 0.93 \\
DiffLinker (given anchors)$^\dagger$ & 0.49 & 3.02 & 0.79 & 93.5 & 32.1 & 68.4 & 88.0 & 0.93 \\
DiffLinker (sampled size)$^\dagger$ & 0.45 & 3.27 & 0.76 & 87.1 & 57.3 & 76.1 & 77.5 & 0.88 \\
DiffLinker (anchors+size)$^\dagger$ & 0.46 & 3.33 & 0.84 & 88.6 & 58.2 & 76.1 & 77.1 & 0.88 \\"""
}

def extract_metrics(summary_path):
    """Loads metrics from the evaluator's output CSV file."""
    if not os.path.exists(summary_path):
        return ["--"] * 8
    try:
        df = pd.read_csv(summary_path)
        row = df.iloc[0]
        qed = f"{row.get('qed', 0.0):.2f}"
        sa = f"{row.get('sa', 0.0):.2f}"
        rings = f"{row.get('rings_n', 0.0):.2f}"
        valid = f"{row.get('validity', 0.0):.1f}"
        unique = f"{row.get('uniqueness', 0.0):.1f}"
        novel = f"{row.get('novelty', 0.0):.1f}"
        rec = f"{row.get('recovery', 0.0):.1f}"
        sc = f"{row.get('sc_rdkit_mean', 0.0):.2f}"
        return [qed, sa, rings, valid, unique, novel, rec, sc]
    except Exception:
        return ["--"] * 8

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_missing', action='store_true', help="If set, automatically runs missing exports/evaluations")
    args = parser.parse_args()

    # Step 1: Run missing evaluations if requested
    for dataset, variants in CONFIGS.items():
        for variant, (output_dir, test_path, train_smi_path) in variants.items():
            summary_path = os.path.join(output_dir, "generated_smiles_summary.csv")
            if not os.path.exists(summary_path) and args.run_missing:
                print(f"\n🔄 Running missing evaluation for {dataset} ({variant})...")
                
                # Run export
                export_cmd = ["python", "export_for_difflinker.py", "--output_dir", output_dir, "--test_path", test_path]
                subprocess.run(export_cmd, check=True)
                
                # Run evaluate
                eval_cmd = [
                    "python", "evaluate_with_difflinker.py",
                    dataset,
                    os.path.join(output_dir, "generated_smiles.txt"),
                    train_smi_path,
                    "4", "True", "None",
                    "datasets/zinc/wehi_pains.csv",
                    "diffusion"
                ]
                subprocess.run(eval_cmd, check=True)

    # Step 2: Compile the final LaTeX table
    latex_output = r"""\begin{table*}[!t]
\centering
\caption{\textbf{Performance comparison on ZINC, CASF-2016, and GEOM-drugs benchmarks.}}
\label{tab:main}

\begingroup
\small
\setlength{\tabcolsep}{2.2pt}
\renewcommand{\arraystretch}{1.0}

\begin{tabular}{lcccccccc}
\hline
\noalign{\vskip 2pt}
Methods & QED$\uparrow$ & SA$\downarrow$ & Rings$\uparrow$ & Valid (\%) & Unique (\%) & Novel (\%) & Recovery (\%) & $\text{SC}_{\text{RDKit}} \uparrow$ \\ 
\noalign{\vskip 2pt}
\hline"""

    for dataset in ['ZINC', 'CASF', 'GEOM']:
        latex_output += f"\n\noalign{{\\vskip 2pt}}\n\\multicolumn{{9}}{{c}}{{\\textbf{{{dataset}}}}} \\\\\n\\noalign{{\\vskip 2pt}}\n\\hline\n"
        
        # Add baselines
        latex_output += BASELINES[dataset] + "\n"
        
        # Extract and format GeoLinker rows
        variants_dict = CONFIGS[dataset]
        labels = {
            'base': 'GeoLinker (base)',
            'anchors': 'GeoLinker (anchors)',
            'sized': 'GeoLinker (sized)',
            'both': 'GeoLinker (both)'
        }
        
        for var, label in labels.items():
            out_dir = variants_dict[var][0]
            summary_file = os.path.join(out_dir, "generated_smiles_summary.csv")
            m = extract_metrics(summary_file)
            
            # Formatting and bolding logic for top values
            row = f"\\textbf{{{label}}} & {m[0]} & {m[1]} & {m[2]} & {m[3]} & {m[4]} & {m[5]} & {m[6]} & {m[7]} \\\\"
            latex_output += row + "\n"
            
        latex_output += "\\hline"

    latex_output += r"""
\end{tabular}
\endgroup

\end{table*}"""

    print("\n" + "="*80)
    print("📋 YOUR COMPILED LATEX TABLE:")
    print("="*80)
    print(latex_output)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()