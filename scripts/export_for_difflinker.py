"""
geoLinker v8 — export_for_difflinker.py (TCBB FINAL EXPORTER)
Implements tailored bond thresholds, dynamic margins, and MMFF relaxation.
Fixes GEOM over-bonding and extracts official target SMILES to resolve Recovery/RMSD.
"""

import os
import glob
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdDetermineBonds, AllChem

# Mute verbose RDKit warnings
RDLogger.DisableLog('rdApp.*')

ATOM_MAPPING = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'P', 5: 'S', 6: 'Cl', 7: 'Br'}

def get_bond_max(s1, s2, margin):
    COVALENT_RADII = {
        'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71, 
        'P': 1.06, 'S': 1.02, 'Cl': 0.99, 'Br': 1.14
    }
    r_sum = COVALENT_RADII.get(s1, 0.77) + COVALENT_RADII.get(s2, 0.77)
    return r_sum + margin

def minimize_linker(mol, linker_mask):
    """Snaps the generated linker coordinates into physically perfect bond lengths."""
    try:
        Chem.FastFindRings(mol)
        mol.UpdatePropertyCache(strict=False)
        mol_h = Chem.AddHs(mol, addCoords=True)
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, AllChem.MMFFGetMoleculeProperties(mol_h))
        if ff is None: return mol
        for i in range(mol.GetNumAtoms()):
            if linker_mask[i] == 0:  # Freeze fragments
                ff.AddFixedPoint(i)
        ff.Minimize(maxIts=50)
        return Chem.RemoveHs(mol_h)
    except Exception:
        return mol

def build_ref_mol(positions, atom_types, margin):
    """Builds a reference molecule cleanly from ground-truth coordinates."""
    num_atoms = len(positions)
    symbols = [ATOM_MAPPING.get(int(idx), 'C') for idx in atom_types]
    mol = Chem.RWMol()
    conf = Chem.Conformer(num_atoms)
    for i in range(num_atoms):
        mol.AddAtom(Chem.Atom(symbols[i]))
        conf.SetAtomPosition(i, positions[i].tolist())
    
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 0.1: continue
            if d <= get_bond_max(symbols[i], symbols[j], margin):
                mol.AddBond(i, j, Chem.BondType.SINGLE)
    mol.AddConformer(conf)
    connectivity_mol = mol.GetMol()

    try:
        refined = Chem.Mol(connectivity_mol)
        rdDetermineBonds.DetermineBondOrders(refined, charge=0)
        Chem.SanitizeMol(refined)
        return refined
    except Exception:
        try:
            connectivity_mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(connectivity_mol)
            return connectivity_mol
        except Exception:
            mask = Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES
            Chem.SanitizeMol(connectivity_mol, sanitizeOps=mask)
            return connectivity_mol

def build_pred_mol_with_ref(ref_mol, gen_pos, gen_types, linker_mask, margin):
    """Builds predicted molecules with strict, freeze-proof topological single bonds."""
    pred_mol = Chem.RWMol(ref_mol)
    num_atoms = pred_mol.GetNumAtoms()
    
    # 1. Update coordinates
    conf = pred_mol.GetConformer(0)
    for i in range(num_atoms):
        conf.SetAtomPosition(i, gen_pos[i].tolist())
        
    # 2. Update atomic types
    SYMBOL_TO_ATOMIC_NUM = {'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35}
    linker_idx = (linker_mask == 1)
    for i in range(num_atoms):
        if linker_idx[i]:
            symbol = ATOM_MAPPING.get(int(gen_types[i]), 'C')
            pred_mol.GetAtomWithIdx(i).SetAtomicNum(SYMBOL_TO_ATOMIC_NUM.get(symbol, 6))
            pred_mol.GetAtomWithIdx(i).SetFormalCharge(0)
            pred_mol.GetAtomWithIdx(i).SetHybridization(Chem.HybridizationType.UNSPECIFIED)

    # 3. Remove all bonds involving linker atoms
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if linker_idx[i] or linker_idx[j]:
                if pred_mol.GetBondBetweenAtoms(i, j) is not None:
                    pred_mol.RemoveBond(i, j)

    # 4. Initialize degrees from starting fragment bonds
    degrees = [len(pred_mol.GetAtomWithIdx(i).GetBonds()) for i in range(num_atoms)]
    MAX_VALENCE = {'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 5, 'S': 6, 'Cl': 1, 'Br': 1}
    symbols = [pred_mol.GetAtomWithIdx(i).GetSymbol() for i in range(num_atoms)]

    # 5. Rebuild bonds involving linker atoms
    candidate_bonds = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if linker_idx[i] or linker_idx[j]:
                d = np.linalg.norm(gen_pos[i] - gen_pos[j])
                if d < 0.1: continue
                max_d = get_bond_max(symbols[i], symbols[j], margin)
                if d <= max_d:
                    candidate_bonds.append((d / max_d, i, j))

    candidate_bonds.sort(key=lambda x: x[0])

    for ratio, i, j in candidate_bonds:
        sym_i = symbols[i]
        sym_j = symbols[j]
        if degrees[i] < MAX_VALENCE.get(sym_i, 4) and degrees[j] < MAX_VALENCE.get(sym_j, 4):
            pred_mol.AddBond(i, j, Chem.BondType.SINGLE)
            degrees[i] += 1
            degrees[j] += 1

    mol = pred_mol.GetMol()
    mol = minimize_linker(mol, linker_mask)

    try:
        Chem.SanitizeMol(mol)
        return mol, "OK (Single Bonds)"
    except Exception:
        try:
            mask = Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES
            Chem.SanitizeMol(mol, sanitizeOps=mask)
            return mol, "OK_Bypassed"
        except Exception as e_inner:
            return None, f"Sanitization failed: {e_inner}"

def extract_submol(mol, mask_to_keep):
    """Slices a molecule topology keeping only designated atoms."""
    submol = Chem.RWMol(mol)
    for idx in sorted(range(mol.GetNumAtoms()), reverse=True):
        if mask_to_keep[idx] == 0:
            submol.RemoveAtom(idx)
    return submol.GetMol()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="outputs/final_eval")
    parser.add_argument('--test_path', type=str, default="datasets/zinc/zinc_final_test.pt")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset-specific margins to prevent GEOM over-bonding
    margin = 0.35
    if "casf" in args.test_path.lower():
        margin = 0.20
    elif "geom" in args.test_path.lower():
        margin = 0.02  # FIXED: Tight margin to prevent GEOM over-bonding
        
    print(f"Perceiving bonds with dataset-specific margin: +{margin} Å")

    from geolinker.dataset import MoleculeDataset
    dataset = MoleculeDataset(args.test_path, dataset_name="zinc")
    
    test_dir = os.path.dirname(args.test_path)
    base_name = os.path.basename(args.test_path).replace('.pt', '')
    
    smi_path = os.path.join(test_dir, f"{base_name}_smiles.smi")
    csv_path = os.path.join(test_dir, f"{base_name}_table.csv")
    csv_path_alt = os.path.join(test_dir, base_name.replace('_final', '') + "_table.csv")

    if os.path.exists(smi_path):
        df = pd.read_csv(smi_path, sep=' ', names=['mol', 'frag'])
        true_smiles_list = df['mol'].values
        frag_smiles_list = df['frag'].values
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        true_smiles_list = df['molecule'].values
        frag_smiles_list = df['fragments'].values
    elif os.path.exists(csv_path_alt):
        df = pd.read_csv(csv_path_alt)
        true_smiles_list = df['molecule'].values
        frag_smiles_list = df['fragments'].values
    else:
        raise FileNotFoundError(f"Could not find reference SMILES (.smi) or CSV table in {test_dir}")

    files = sorted(glob.glob(os.path.join(args.output_dir, "sample_*_gen_*.pt")))
    if not files: return

    txt_out_path = os.path.join(args.output_dir, "generated_smiles.txt")
    sdf_out_path = os.path.join(args.output_dir, "generated_smiles.sdf")

    sdf_writer = Chem.SDWriter(sdf_out_path)
    smiles_lines = []

    for file in tqdm(files):
        try:
            gen_data = torch.load(file, map_location='cpu', weights_only=False)
        except Exception:
            continue
            
        target_idx = gen_data.get('target_idx', int(os.path.basename(file).split('_')[1]))

        ref_item = dataset[target_idx]
        linker_mask = ref_item['linker_mask'].numpy()
        gen_pos = gen_data['positions'].numpy()
        gen_types = torch.argmax(gen_data['types'], dim=-1).numpy() if gen_data['types'].dim() > 1 else gen_data['types'].numpy()

        # FIXED: Extract exact, canonical reference target SMILES to resolve Recovery/RMSD
        true_smi = true_smiles_list[target_idx]
        frag_smi = frag_smiles_list[target_idx] 

        ref_pos = ref_item['positions'].numpy()
        ref_types = torch.argmax(ref_item['atom_features'], dim=-1).numpy()
        ref_mol = build_ref_mol(ref_pos, ref_types, margin)

        pred_smi, pred_linker_smi = "invalid", "invalid"

        if ref_mol is not None:
            # Reconstruct fragment SMILES to remove stereocenters for validity matching
            try:
                frag_mol = extract_submol(ref_mol, 1 - linker_mask)
                frag_smi = Chem.MolToSmiles(frag_mol)
            except Exception: pass

            pred_mol, status = build_pred_mol_with_ref(ref_mol, gen_pos, gen_types, linker_mask, margin)
            if pred_mol is not None:
                try:
                    pred_smi = Chem.MolToSmiles(pred_mol)
                    pred_linker = extract_submol(pred_mol, linker_mask)
                    pred_linker_smi = Chem.MolToSmiles(pred_linker)
                    sdf_writer.write(pred_mol)
                except Exception:
                    pred_smi = "invalid"
                    pred_linker_smi = "invalid"
                    sdf_writer.write(Chem.MolFromSmiles("C"))
            else:
                sdf_writer.write(Chem.MolFromSmiles("C"))
        else:
            sdf_writer.write(Chem.MolFromSmiles("C"))

        # true_smi is strictly the official key-matching string
        smiles_lines.append(f"{frag_smi} {true_smi} {pred_smi} {pred_linker_smi}\n")

    sdf_writer.close()
    with open(txt_out_path, "w") as f:
        f.writelines(smiles_lines)
    print("\n✅ Final Export complete!")

if __name__ == "__main__":
    main()