#!/usr/bin/env python

import os
import csv
import numpy as np
import pandas as pd
import sys
import networkx as nx

from networkx.algorithms import isomorphism
from rdkit import Chem
from rdkit.Chem import MolStandardize, QED, rdMolAlign, rdMolDescriptors
from src.delinker_utils import calc_SC_RDKit, frag_utils, sascorer
from src.utils import disable_rdkit_logging
from tqdm import tqdm

disable_rdkit_logging()

if len(sys.argv) != 9:
    print("Not provided all arguments")
    quit()

data_set = sys.argv[1]  
gen_smi_file = sys.argv[2]  
train_set_path = sys.argv[3]  
n_cores = int(sys.argv[4])  
verbose = bool(sys.argv[5])  
if sys.argv[6] == "None":
    restrict = None
else:
    restrict = int(sys.argv[6])  
pains_smarts_loc = sys.argv[7]  
method = sys.argv[8]

data = []
with open(gen_smi_file, 'r') as f:
    for line in tqdm(f.readlines()):
        parts = line.strip().split(' ')
        data.append({
            'fragments': parts[0],
            'true_molecule': parts[1],
            'pred_molecule': parts[2],
            'pred_linker': parts[3] if len(parts) > 3 else '',
        })

if restrict is not None:
    data = data[:restrict]

summary = {}

# -------------- Validity -------------- #
def is_valid(pred_mol_smiles, frag_smiles):
    pred_mol = Chem.MolFromSmiles(pred_mol_smiles)
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None or pred_mol is None: return False
    try:
        Chem.SanitizeMol(pred_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except: return False
    if len(pred_mol.GetSubstructMatch(frag)) != frag.GetNumAtoms(): return False
    return True

valid_cnt, total_cnt = 0, 0
for obj in tqdm(data):
    valid = is_valid(obj['pred_molecule'], obj['fragments'])
    obj['valid'] = valid
    valid_cnt += valid
    total_cnt += 1

validity = valid_cnt / total_cnt * 100
print(f'Validity: {validity:.3f}%')
summary['validity'] = validity

# ----------------- QED ------------------ #
qed_values = []
for obj in tqdm(data):
    if not obj['valid']: continue
    qed = QED.qed(Chem.MolFromSmiles(obj['pred_molecule']))
    obj['qed'] = qed
    qed_values.append(qed)
print(f'Mean QED: {np.mean(qed_values):.3f}')

# ----------------- SA ------------------ #
sa_values = []
for obj in tqdm(data):
    if not obj['valid']: continue
    sa = sascorer.calculateScore(Chem.MolFromSmiles(obj['pred_molecule']))
    obj['sa'] = sa
    sa_values.append(sa)
print(f'Mean SA: {np.mean(sa_values):.3f}')

# ----------------- Number of Rings ------------------ #
rings_n_values = []
for obj in tqdm(data):
    if not obj['valid']: continue
    try: rings_n = rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(obj['pred_linker']))
    except: continue
    obj['rings_n'] = rings_n
    rings_n_values.append(rings_n)
print(f'Mean Number of Rings: {np.mean(rings_n_values):.3f}')

# -------------- Uniqueness -------------- #
true2samples = dict()
for obj in tqdm(data):
    if not obj['valid']: continue
    key = f"{obj['true_molecule']}_{obj['fragments']}"
    true2samples.setdefault(key, []).append(obj['pred_molecule'])

unique_cnt, total_cnt = 0, 0
for samples in tqdm(true2samples.values()):
    unique_cnt += len(set(samples))
    total_cnt += len(samples)
uniqueness = unique_cnt / total_cnt * 100
print(f'Uniqueness: {uniqueness:.3f}%')

# ----------------- Novelty ---------------- #
linkers_train = set()
with open(train_set_path, 'r') as f:
    for line in f: linkers_train.add(line.strip())

novel_cnt, total_cnt = 0, 0
for obj in tqdm(data):
    if not obj['valid']: continue
    try:
        linker = Chem.RemoveStereochemistry(obj['pred_linker'])
        linker = MolStandardize.canonicalize_tautomer_smiles(Chem.MolToSmiles(linker))
    except: linker = obj['pred_linker']
    novel = linker not in linkers_train
    obj['novel'] = novel
    novel_cnt += novel
    total_cnt += 1
novelty = novel_cnt / total_cnt * 100
print(f'Novelty: {novelty:.3f}%')

# ----------------- Strict SMILES Recovery ---------------- #
recovered_inputs = set()
all_inputs = set()
for obj in tqdm(data):
    if not obj['valid']:
        obj['recovered'] = False
        continue
    key = obj['true_molecule'] + '_' + obj['fragments']
    try:
        true_mol = Chem.MolFromSmiles(obj['true_molecule'])
        Chem.RemoveStereochemistry(true_mol)
        true_mol_smi = Chem.MolToSmiles(Chem.RemoveHs(true_mol))
    except:
        true_mol = Chem.MolFromSmiles(obj['true_molecule'], sanitize=False)
        Chem.RemoveStereochemistry(true_mol)
        true_mol_smi = Chem.MolToSmiles(Chem.RemoveHs(true_mol, sanitize=False))

    pred_mol = Chem.MolFromSmiles(obj['pred_molecule'])
    Chem.RemoveStereochemistry(pred_mol)
    pred_mol_smi = Chem.MolToSmiles(Chem.RemoveHs(pred_mol))

    recovered = true_mol_smi == pred_mol_smi
    obj['recovered'] = recovered
    if recovered: recovered_inputs.add(key)
    all_inputs.add(key)
recovery = len(recovered_inputs) / len(all_inputs) * 100 if all_inputs else 0.0
print(f'Strict SMILES Recovery: {recovery:.3f}%')

# ----------------- PAINS Filter ---------------- #
def check_pains(mol, pains):
    for pain in pains:
        if mol.HasSubstructMatch(pain): return False
    return True

with open(pains_smarts_loc, 'r') as f:
    pains_smarts = set([Chem.MolFromSmarts(line[0], mergeHs=True) for csv_line in csv.reader(f) if csv_line for line in [csv_line]])

passed_pains_cnt, total_cnt = 0, 0
for obj in tqdm(data):
    if not obj['valid']: continue
    pred_mol = Chem.MolFromSmiles(obj['pred_molecule'])
    passed_pains = check_pains(pred_mol, pains_smarts)
    passed_pains_cnt += passed_pains
    total_cnt += 1
print(f'Passed PAINS: {passed_pains_cnt / total_cnt * 100:.3f}%')

# ----------------- RA Filter ---------------- #
def check_ring_filter(linker):
    ssr = Chem.GetSymmSSSR(linker)
    for ring in ssr:
        for atom_idx in ring:
            for bond in linker.GetAtomWithIdx(atom_idx).GetBonds():
                if bond.GetBondType() == 2 and bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring:
                    return False
    return True

passed_ring_filter_cnt, total_cnt = 0, 0
for obj in tqdm(data):
    if not obj['valid']: continue
    try: passed_ring_filter = check_ring_filter(Chem.MolFromSmiles(obj['pred_linker'], sanitize=False))
    except: passed_ring_filter = False
    passed_ring_filter_cnt += passed_ring_filter
    total_cnt += 1
print(f'Passed Ring Filter: {passed_ring_filter_cnt / total_cnt * 100:.3f}%')

# ----------------------- PURE TOPOLOGICAL RMSD --------------------- #
def get_pure_topology(mol):
    """Creates a graph based ONLY on atomic numbers and connectivity."""
    G = nx.Graph()
    if mol is None: return G
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), element=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G

sdf_path = gen_smi_file[:-3] + 'sdf'
pred_mol_3d = Chem.SDMolSupplier(sdf_path)

if data_set == 'ZINC':
    base_path = 'datasets/zinc/zinc_final_test'
elif data_set == 'CASF':
    base_path = 'datasets/casf/casf_final_test'
elif data_set == 'GEOM':
    base_path = 'datasets/geom/geom_multifrag_test'

# Handle naming variations from Zenodo
mol_path = f"{base_path}_molecules.sdf"
if not os.path.exists(mol_path): mol_path = f"{base_path}_mol.sdf"
if not os.path.exists(mol_path): mol_path = mol_path.replace('_final', '')

smi_path = f"{base_path}_smiles.smi"
csv_path = f"{base_path}_table.csv"
csv_path_alt = csv_path.replace('_final', '')

if os.path.exists(smi_path):
    true_smi = pd.read_csv(smi_path, sep=' ', names=['mol', 'frag'])['mol'].values
elif os.path.exists(csv_path):
    true_smi = pd.read_csv(csv_path)['molecule'].values
elif os.path.exists(csv_path_alt):
    true_smi = pd.read_csv(csv_path_alt)['molecule'].values

true_mol_3d = list(Chem.SDMolSupplier(mol_path))
true_smi2mol3d = dict(zip(true_smi, true_mol_3d))

rmsd_list = []
topological_recovery_cnt = 0
all_valid_inputs = set()

node_match = isomorphism.categorical_node_match('element', 0)

for i, (obj, pred) in tqdm(enumerate(zip(data, pred_mol_3d)), total=len(data)):
    if not obj.get('valid', False): continue
    
    key = obj['true_molecule'] + '_' + obj['fragments']
    all_valid_inputs.add(key)

    # 1. Primary Lookup (Exact SMILES matching)
    true = true_smi2mol3d.get(obj['true_molecule'], None)
    if true is None:
        try:
            norm = Chem.MolToSmiles(Chem.MolFromSmiles(obj['true_molecule']))
            true = true_smi2mol3d.get(norm, None)
        except: pass
    # 2. Secondary Fallback (Clamped indexing)
    if true is None:
        samples_per_target = max(len(data) // len(true_mol_3d), 1) if len(true_mol_3d) > 0 else 1
        target_idx = min(i // samples_per_target, len(true_mol_3d) - 1)
        true = true_mol_3d[target_idx]

    if pred is None or true is None:
        continue

    Chem.RemoveStereochemistry(true)
    true = Chem.RemoveHs(true)
    Chem.RemoveStereochemistry(pred)
    pred = Chem.RemoveHs(pred)

    G1 = get_pure_topology(pred)
    G2 = get_pure_topology(true)
    GM = isomorphism.GraphMatcher(G1, G2, node_match=node_match)
    
    if GM.is_isomorphic():
        topological_recovery_cnt += 1
        try:
            error = Chem.rdMolAlign.GetBestRMS(pred, true)
            frag_size = Chem.MolFromSmiles(obj['fragments']).GetNumAtoms()
            num_linker = pred.GetNumAtoms() - frag_size
            num_atoms = pred.GetNumAtoms()
            if num_linker > 0:
                error *= np.sqrt(num_atoms / num_linker)
            rmsd_list.append(error)
            obj['rmsd'] = error
        except Exception:
            pass

topo_recovery = (topological_recovery_cnt / len(all_valid_inputs)) * 100 if all_valid_inputs else 0.0
print(f'Topological Recovery: {topo_recovery:.3f}%')
print(f'Mean RMSD: {np.mean(rmsd_list) if rmsd_list else 0.0:.3f}')

# ----------------------------- SC-RDKit -------------------------- #
def calc_sc_rdkit_full_mol(gen_mol, ref_mol):
    try:
        _ = rdMolAlign.GetO3A(gen_mol, ref_mol).Align()
        return calc_SC_RDKit.calc_SC_RDKit_score(gen_mol, ref_mol)
    except: return -0.5

sc_rdkit_list = []
for i, (obj, pred) in tqdm(enumerate(zip(data, pred_mol_3d)), total=len(data)):
    if not obj.get('valid', False): continue

    # 1. Primary Lookup (Exact SMILES matching)
    true = true_smi2mol3d.get(obj['true_molecule'], None)
    if true is None:
        try:
            norm = Chem.MolToSmiles(Chem.MolFromSmiles(obj['true_molecule']))
            true = true_smi2mol3d.get(norm, None)
        except: pass
    # 2. Secondary Fallback (Clamped indexing)
    if true is None:
        samples_per_target = max(len(data) // len(true_mol_3d), 1) if len(true_mol_3d) > 0 else 1
        target_idx = min(i // samples_per_target, len(true_mol_3d) - 1)
        true = true_mol_3d[target_idx]

    if pred is None or true is None:
        continue

    score = calc_sc_rdkit_full_mol(pred, true)
    sc_rdkit_list.append(score)

sc_rdkit_list = np.array(sc_rdkit_list) if sc_rdkit_list else np.array([])
print(f'SC_RDKit > 0.7: {(sc_rdkit_list > 0.7).sum() / len(sc_rdkit_list) * 100 if len(sc_rdkit_list) else 0.0:.3f}%')
print(f'SC_RDKit > 0.8: {(sc_rdkit_list > 0.8).sum() / len(sc_rdkit_list) * 100 if len(sc_rdkit_list) else 0.0:.3f}%')
print(f'SC_RDKit > 0.9: {(sc_rdkit_list > 0.9).sum() / len(sc_rdkit_list) * 100 if len(sc_rdkit_list) else 0.0:.3f}%')
print(f'Mean SC_RDKit: {np.mean(sc_rdkit_list) if len(sc_rdkit_list) else -0.5:.3f}')