"""
geoLinker v8 — train.py (MULTI-CORE OPTIMIZED FOR WINDOWS)
"""

from html import parser
import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import numpy as np
import random

from geolinker.dataset import MoleculeDataset, collate_fn
from geolinker.models import get_model
from geolinker.diffusion import EquivariantDiffusion
from geolinker.losses import UnifiedLoss


class MaxAtomsBatchSampler(Sampler):
    """
    Yields batches of dataset indices bounded by TOTAL ATOM COUNT, not a
    fixed molecule count. A fixed --batch_size lets total atoms per batch
    swing unpredictably depending on which molecules get randomly drawn
    together -- most batches are fine, but an unlucky combination of several
    large molecules can spike peak memory far past what a "typical" batch
    needs. Since EGNN_Attention_Layer's memory scales with each molecule's
    own atom count, that's both the root cause of the allocator fragmentation
    we've been fighting AND, as seen directly, capable of causing an outright
    CUDA OOM on its own. Capping the atom budget per batch bounds the
    worst case directly, rather than mitigating it after the fact.
    """
    def __init__(self, dataset, max_atoms_per_batch: int = 650, shuffle: bool = True):
        self.max_atoms_per_batch = max_atoms_per_batch
        self.shuffle = shuffle
        # Atom count per molecule, computed once up front (cheap: just len()
        # on already-loaded data, no tensor ops).
        self.sizes = [len(dataset.data[i]['positions']) for i in range(len(dataset))]
        oversized = [i for i, s in enumerate(self.sizes) if s > max_atoms_per_batch]
        if oversized:
            print(f"  ⚠️  {len(oversized)} molecule(s) alone exceed max_atoms_per_batch="
                  f"{max_atoms_per_batch} (e.g. index {oversized[0]} has {self.sizes[oversized[0]]} atoms) "
                  f"-- they'll each run alone in their own batch.")

    def __iter__(self):
        indices = list(range(len(self.sizes)))
        if self.shuffle:
            random.shuffle(indices)
        batch, total = [], 0
        for idx in indices:
            n = self.sizes[idx]
            if batch and total + n > self.max_atoms_per_batch:
                yield batch
                batch, total = [], 0
            batch.append(idx)
            total += n
        if batch:
            yield batch

    def __len__(self):
        total_atoms = sum(self.sizes)
        return max(1, -(-total_atoms // self.max_atoms_per_batch))  # ceil division

def parse_args():
    parser = argparse.ArgumentParser(description="Train Unified B-END Model")
    parser.add_argument('--train_path', type=str, default=r"D:\research_datasets\GeoLinker\zinc\zinc_final_train.pt")
    parser.add_argument('--val_path', type=str, default=r"D:\research_datasets\GeoLinker\zinc\zinc_final_val.pt")
    parser.add_argument('--subset_size', type=int, default=None, help="Train on a small subset first for quick testing")
    
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32) # Default reduced to 32 to maximize GPU speedups
    parser.add_argument('--max_atoms_per_batch', type=int, default=None,
                         help="If set, batches are built by TOTAL ATOM COUNT instead of "
                              "molecule count (overrides --batch_size for the train loader). "
                              "Bounds worst-case per-batch memory directly -- recommended over "
                              "--batch_size once you've hit OOM/fragmentation issues.")
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    parser.add_argument('--save_dir', type=str, default=r"D:\research_datasets\GeoLinker\checkpoints")
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n============================================================")
    print(f"TRAINING UNIFIED B-END MODEL")
    print(f"============================================================")
    print(f"Device:            {device}")
    print(f"Subset Size:       {args.subset_size}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"============================================================\n")

    # 1. Load Standardized Datasets
    train_dataset = MoleculeDataset(args.train_path, dataset_name="zinc")
    val_dataset = MoleculeDataset(args.val_path, dataset_name="zinc")

    if args.subset_size:
        train_dataset.data = train_dataset.data[:args.subset_size]
        print(f"✂️ Sliced training dataset down to {len(train_dataset):,} samples for testing.")

    # 2. Dataloaders (MULTI-CORE SPEEDUPS ENABLED)
    if args.max_atoms_per_batch:
        print(f"Using atom-budget batching: max {args.max_atoms_per_batch} atoms/batch (train set only)")
        train_sampler = MaxAtomsBatchSampler(train_dataset, max_atoms_per_batch=args.max_atoms_per_batch, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0,  # Fixes Windows spawn overhead
            pin_memory=True
        )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    # 3. Model & Diffusion
    model = get_model(variant='unified', num_atom_features=8, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    diffusion_model = EquivariantDiffusion(model=model, timesteps=1000, beta_schedule='cosine', variant='unified').to(device)
    loss_fn = UnifiedLoss().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"Resuming from checkpoint {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', ckpt.get('val_loss', float('inf')))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = diffusion_model.training_step(batch)
            loss, loss_dict = loss_fn(outputs, batch)
            
            if torch.isnan(loss) or torch.isinf(loss) or loss > 100.0:
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'clash': f"{loss_dict.get('clash', 0):.3f}",
                'endpt': f"{loss_dict.get('endpoint', 0):.3f}",
                'repel': f"{loss_dict.get('repulsion', 0):.3f}"
            })

            # Windows/WDDM-specific fragmentation mitigation: every batch has a
            # different total atom count (molecules vary in size and get
            # randomly grouped), so the allocator keeps requesting new block
            # shapes it can't cleanly reuse. `expandable_segments` -- the
            # normal fix for this -- isn't available on Windows, and clearing
            # the cache only once per epoch is too infrequent to catch this;
            # the collapse happens mid-epoch. Clearing periodically (not every
            # batch -- that would hurt throughput) keeps reserved memory from
            # ratcheting up faster than it can be reclaimed.
            if torch.cuda.is_available() and batch_idx > 0 and batch_idx % 250 == 0:
                torch.cuda.empty_cache()

        avg_train_loss = np.mean(train_losses)
        scheduler.step()

        # Validation Step (loss + linker CoM-drift diagnostic in one pass --
        # the diagnostic uses the model's one-shot x_0 estimate at a random
        # timestep, so it's ~free once we're already forward-passing anyway)
        model.eval()
        val_losses = []
        com_drifts = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = diffusion_model.training_step(batch)
                loss, _ = loss_fn(outputs, batch)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_losses.append(loss.item())

                est_pos = outputs['est_positions']
                linker_mask = batch['linker_mask'].bool()
                batch_mask = batch['batch_mask']
                if linker_mask.sum() > 0:
                    for b in batch_mask.unique():
                        sel = linker_mask & (batch_mask == b)
                        if sel.sum() > 0:
                            drift = est_pos[sel].mean(dim=0).norm().item()
                            com_drifts.append(drift)

        avg_val_loss = np.mean(val_losses) if val_losses else float('nan')
        avg_com_drift = np.mean(com_drifts) if com_drifts else float('nan')

        print(f"Epoch {epoch+1} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Linker CoM Drift: {avg_com_drift:.3f} Å")
        if avg_com_drift > 5.0:
            print(f"  ⚠️  Linker CoM drift is still large (>5 Å). If this isn't trending down by "
                  f"epoch {min(5, args.epochs)}, stop and re-check the noise pipeline before "
                  f"spending time on a full generate.py run.")

        # Save Best Model
        if avg_val_loss < best_val_loss or epoch == start_epoch:
            best_val_loss = avg_val_loss
            best_path = os.path.join(args.save_dir, "best_unified.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, best_path)
            print(f"  🌟 Saved best checkpoint to {best_path}")

        # GPU memory diagnostic + cache release. The EGNN layer builds a full
        # [total_batch_atoms, total_batch_atoms] pairwise tensor and masks out
        # cross-molecule pairs -- since batch atom count varies with which
        # molecules get randomly drawn together, an unlucky large-molecule
        # batch can spike peak memory, and PyTorch's caching allocator doesn't
        # release that reservation on its own. Printing allocated/reserved
        # here makes that growth visible epoch-to-epoch instead of only
        # discovering it hours later as a mysterious slowdown; empty_cache()
        # gives the allocator a chance to defragment between epochs.
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU memory -- allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")
            torch.cuda.empty_cache()

    print("\n============================================================")
    print("🎉 TRAINING COMPLETE!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("============================================================\n")

if __name__ == "__main__":
    main()