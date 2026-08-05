"""
geoLinker v8 — generate.py (FINAL MULTI-SAMPLE SAMPLER)
"""

import os
import argparse
import torch
from tqdm import tqdm

from geolinker.dataset import MoleculeDataset
from geolinker.models import get_model
from geolinker.diffusion import EquivariantDiffusion

def parse_args():
    parser = argparse.ArgumentParser(description="Generate molecules using Unified B-END")
    parser.add_argument('--variant', type=str, default='both', choices=['base', 'anchor', 'sized', 'both'])
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--fragments', type=str, required=True, help="Path to test dataset (.pt file)")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save generated .pt samples")
    parser.add_argument('--n_samples', type=int, default=100, help="Number of targets to evaluate")
    parser.add_argument('--samples_per_target', type=int, default=10, help="Number of samples per target")
    parser.add_argument('--num_steps', type=int, default=100, help="Number of SDE steps")
    parser.add_argument('--strategy', type=str, default='ddpm', choices=['ddpm', 'ddim'])
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = MoleculeDataset(args.fragments, dataset_name="zinc")
    
    print(f"Loading model from {args.checkpoint}...")
    model = get_model(args.variant, num_atom_features=8, hidden_dim=128, num_layers=6).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    diffusion = EquivariantDiffusion(model=model, timesteps=1000, beta_schedule='cosine', variant=args.variant).to(device)

    targets_to_run = min(args.n_samples, len(dataset))
    print(f"Generating for {targets_to_run} targets ({args.samples_per_target} samples/target) using {args.strategy.upper()} ({args.num_steps} steps)...")
    
    for i in tqdm(range(targets_to_run), desc="Target Molecules"):
        data_item = dataset[i]
        
        for s_idx in range(args.samples_per_target):
            with torch.no_grad():
                final_pos, final_types = diffusion.sample(data_item=data_item, num_steps=args.num_steps, sample_strategy=args.strategy)
                
            types_one_hot = torch.nn.functional.one_hot(final_types, num_classes=8).float()
            
            output_data = {
                'positions': final_pos.cpu(),
                'types': types_one_hot.cpu(),
                'linker_mask': data_item['linker_mask'].cpu(),
                'target_idx': i  # Save exact index for foolproof alignment
            }
            torch.save(output_data, os.path.join(args.output_dir, f"sample_{i:04d}_gen_{s_idx:02d}.pt"))

if __name__ == "__main__":
    main()