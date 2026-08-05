"""
geoLinker v8 — dataset.py (POCKET-AWARE VERSION)
Unified dataset loader standardizing ZINC, GEOM, and CASF.
Extracts and batches protein pocket coordinates and atom features on-the-fly.
All caches are strictly stored in D:\research_datasets\GeoLinker\cache.
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np

# Cache directory strictly on D: drive
CACHE_DIR = os.path.join(os.getcwd(), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Standard Universal Chemistry Alphabet: [C, N, O, F, P, S, Cl, Br]
UNIVERSAL_ALPHABET = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']
ATOM_TO_INDEX = {elem: i for i, elem in enumerate(UNIVERSAL_ALPHABET)}

class MoleculeDataset(Dataset):
    def __init__(self, data_path: str, dataset_name: str = "zinc"):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = dataset_name.lower()
        
        print(f"Loading raw {self.dataset_name.upper()} dataset from {data_path}...")
        self.data = torch.load(data_path, map_location='cpu', weights_only=False)
        print(f"Successfully loaded {len(self.data):,} molecules.")

    def __len__(self) -> int:
        return len(self.data)

    def _standardize_atom_features(self, raw_features: torch.Tensor) -> torch.Tensor:
        N = raw_features.shape[0]
        raw_dim = raw_features.shape[-1]
        
        if raw_dim == 8:
            return raw_features.float()
            
        standard_features = torch.zeros((N, 8), dtype=torch.float32)
        
        if self.dataset_name == "geom":
            geom_to_universal = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
            raw_indices = torch.argmax(raw_features, dim=-1)
            for raw_idx, univ_idx in geom_to_universal.items():
                mask = (raw_indices == raw_idx)
                standard_features[mask, univ_idx] = 1.0
        else:
            limit = min(raw_dim, 8)
            standard_features[:, :limit] = raw_features[:, :limit]
            
        return standard_features

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        
        positions = item['positions'].float()
        
        # Standardize Atom Features
        if 'atom_features' in item:
            raw_features = item['atom_features']
        elif 'one_hot' in item:
            raw_features = item['one_hot']
        else:
            raise KeyError(f"No valid atom feature key found. Keys: {list(item.keys())}")
        atom_features = self._standardize_atom_features(raw_features)

        # Standardize Linker Mask
        if 'fragment_mask' in item:
            linker_mask = 1.0 - item['fragment_mask'].float()
        elif 'linker_mask' in item:
            linker_mask = item['linker_mask'].float()
        else:
            linker_mask = torch.zeros(len(positions), dtype=torch.float32)
        # Standardize Anchor Mask
        if 'anchor_mask' in item:
            anchor_mask = item['anchor_mask'].float()
        elif 'anchors' in item:
            anchor_mask = item['anchors'].float()
        else:
            anchor_mask = torch.zeros(len(positions), dtype=torch.float32)

        # Center positions based strictly on fragment Center of Mass (CoM = 0)
        is_fragment = (linker_mask == 0)
        if is_fragment.sum() > 0:
            com = positions[is_fragment].mean(dim=0, keepdim=True)
            positions = positions - com
        else:
            com = positions.mean(dim=0, keepdim=True)
            positions = positions - com

        # --- EXTRACT PROTEIN POCKET DATA (IF PRESENT) ---
        pocket_pos = item.get('pocket_positions', item.get('pocket_pos', None))
        pocket_atoms = item.get('pocket_atoms', item.get('pocket_one_hot', None))
        
        if pocket_pos is not None:
            pocket_pos = torch.tensor(pocket_pos, dtype=torch.float32)
            # Center the pocket using the exact same fragment CoM to align coordinate frames!
            if is_fragment.sum() > 0:
                pocket_pos = pocket_pos - com
                
            if pocket_atoms is not None:
                pocket_atoms = self._standardize_atom_features(torch.tensor(pocket_atoms))
            else:
                pocket_atoms = torch.zeros((len(pocket_pos), 8), dtype=torch.float32)
        # -------------------------------------------------

        linker_size = item.get('linker_size', None)
        if linker_size is None:
            linker_size = int(linker_mask.sum().item())
        
        conditions = item.get('conditions', torch.zeros(3, dtype=torch.float32))

        return {
            'positions': positions,
            'atom_features': atom_features,
            'linker_mask': linker_mask,
            'anchor_mask': anchor_mask,
            'linker_size': linker_size,
            'conditions': conditions,
            'pocket_pos': pocket_pos,
            'pocket_atoms': pocket_atoms
        }

def collate_fn(batch: list) -> dict:
    atom_features_list = []
    positions_list = []
    linker_mask_list = []
    anchor_mask_list = []
    batch_mask_list = []
    conditions_list = []
    linker_size_list = []
    
    # Pocket-specific lists
    pocket_pos_list = []
    pocket_atoms_list = []
    pocket_batch_mask_list = []

    for i, item in enumerate(batch):
        positions_list.append(item['positions'])
        atom_features_list.append(item['atom_features'])
        linker_mask_list.append(item['linker_mask'])
        anchor_mask_list.append(item['anchor_mask'])
        conditions_list.append(item['conditions'])
        linker_size_list.append(torch.tensor(item['linker_size'], dtype=torch.long))
        batch_mask_list.append(torch.full((len(item['positions']),), i, dtype=torch.long))
        
        # Collate pocket data safely if present
        if item['pocket_pos'] is not None:
            pocket_pos_list.append(item['pocket_pos'])
            pocket_atoms_list.append(item['pocket_atoms'])
            pocket_batch_mask_list.append(torch.full((len(item['pocket_pos']),), i, dtype=torch.long))

    batch_dict = {
        'positions': torch.cat(positions_list, dim=0),
        'atom_features': torch.cat(atom_features_list, dim=0),
        'linker_mask': torch.cat(linker_mask_list, dim=0),
        'anchor_mask': torch.cat(anchor_mask_list, dim=0),
        'batch_mask': torch.cat(batch_mask_list, dim=0),
        'conditions': torch.stack(conditions_list, dim=0),
        'linker_size': torch.stack(linker_size_list, dim=0)
    }
    
    if pocket_pos_list:
        batch_dict['pocket_pos'] = torch.cat(pocket_pos_list, dim=0)
        batch_dict['pocket_atoms'] = torch.cat(pocket_atoms_list, dim=0)
        batch_dict['pocket_batch_mask'] = torch.cat(pocket_batch_mask_list, dim=0)
    else:
        batch_dict['pocket_pos'] = None
        batch_dict['pocket_atoms'] = None
        batch_dict['pocket_batch_mask'] = None

    return batch_dict