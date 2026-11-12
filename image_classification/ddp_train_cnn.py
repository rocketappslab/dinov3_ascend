import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import time
from datetime import timedelta
import numpy as np
from capsule_data import NANFANGDataset
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    Resized,
    Resize
)
from monai.networks.nets import DenseNet121, resnet50, resnet101, resnet152
import time
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import torchvision

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_npu")

def setup(rank, world_size):
    """Initialize the distributed environment."""
    # Init process group
    dist.init_process_group(
        backend="hccl",  # NCCL is optimized for NVIDIA GPUs
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180)
    )
    
    # Set device for this process
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DISTRIBUTEDDATAPARALLEL TRAINING (Using {world_size} GPUs)")
        print(f"{'='*60}\n")
    
    print(f"Rank {rank} initialized on {device}")
    
    transform_train = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            Resize(spatial_size=(224, 224), mode="bilinear")
        ]
    )
    OUT_CHANNELS = 35
    GBS=4096
    EXP_NAME = "exp_v2plusv1sb_resnet101"
    CKPT_DIR = f"checkpoints/{EXP_NAME}"
    print("loading NANFANGDataset...")
    train_csv = '/home/ma-user/work/rocket/Dataset/nanfang_capsule/preprocessed_oriname/v2_plus_v1sb_train.csv'
    image_root = '/home/ma-user/work/rocket/Dataset/nanfang_capsule/v2_plus_v1sb'
    train_dataset = NANFANGDataset(train_csv, image_root, transform_train)
    

    # Distributed sampler splits data across GPUs
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    # Each GPU gets batch_size // world_size samples
    batch_size_per_gpu = GBS // world_size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size_per_gpu, 
        sampler=train_sampler,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Model setup
    
    # model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=OUT_CHANNELS).to(device)
    
    # resnet
    model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
    # model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, OUT_CHANNELS)
    model = model.to(device)

    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    use_amp = torch.npu.is_available()
    scaler = torch.amp.GradScaler("npu", enabled=use_amp)

    num_epochs = 100
    log_interval = 10  # Log loss every 10 steps

    if rank == 0:
        start_time = time.time()

    # --- NEW: Initialize TensorBoard and best metric tracker on rank 0 ---
    if rank == 0:
        writer = SummaryWriter(log_dir=f"./runs/{EXP_NAME}")
        best_acc = 0.0
        os.makedirs(CKPT_DIR, exist_ok=True) # Ensure save directory exists

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        
        if torch.npu.is_available():
            torch.npu.synchronize()
        if rank == 0:
            epoch_start = time.time()
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs}", 
            disable=(rank != 0),
            dynamic_ncols=True
        )
        
        for batch_idx, (data, target, path) in enumerate(pbar):
            # import pdb;pdb.set_trace()
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("npu", enabled=use_amp):
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            bs = data.size(0)
            step_loss = loss.item()
            total_loss += step_loss * bs
            _, predicted = output.max(1)
            total += bs
            correct += predicted.eq(target).sum().item()
            
            # Print every log_interval steps on rank 0
            step = batch_idx + 1
            if rank == 0 and step % log_interval == 0:
                running_loss = total_loss / total
                running_acc = 100. * correct / total
                
                # --- NEW: Log step metrics to TensorBoard ---
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/Batch_Loss', step_loss, global_step)
                writer.add_scalar('Train/Running_Loss', running_loss, global_step)
                writer.add_scalar('Train/Running_Acc', running_acc, global_step)
                
                tqdm.write(
                    f"[Epoch {epoch+1}/{num_epochs} | Step {step}/{len(train_loader)}] "
                    f"Batch Loss: {step_loss:.4f} | Running Loss: {running_loss:.4f} | Running Acc: {running_acc:.2f}%"
                )
                
            if rank == 0:
                pbar.set_postfix(loss=f"{total_loss / total:.4f}")
        
        # Aggregate global metrics across all GPUs/NPUs
        metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        global_loss, global_correct, global_total = metrics.tolist()
        
        if torch.npu.is_available():
            torch.npu.synchronize()
            
        if rank == 0:
            epoch_time = time.time() - epoch_start
            avg_loss = global_loss / global_total
            accuracy = 100. * global_correct / global_total
            
            # --- NEW: Log epoch metrics to TensorBoard ---
            writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
            writer.add_scalar('Train/Epoch_Acc', accuracy, epoch)
            
            tqdm.write(
                f"\n[Epoch {epoch+1} Complete] Time: {epoch_time:.2f}s | "
                f"Global Loss: {avg_loss:.4f} | Global Acc: {accuracy:.2f}%"
            )
            
            # --- NEW: Save best model weights ---
            if accuracy > best_acc and accuracy > 90:
                best_acc = accuracy
                tqdm.write(f"*** New best accuracy: {best_acc:.2f}%! Saving model... ***\n")
                
                # Handle DistributedDataParallel unwrapping
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                
                # Save the checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, f"{CKPT_DIR}/best_model.pth")

    # --- NEW: Close TensorBoard writer ---
    if rank == 0:
        writer.close()

    if rank == 0:
        total_time = time.time() - start_time
        print(f"Total Training Time: {total_time:.2f}s")
    dist.destroy_process_group()

if __name__ == "__main__":
    # torchrun sets these environment variables automatically
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    setup(local_rank, world_size)
