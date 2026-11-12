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
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import torchvision

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_npu")


from capsule_data import LABEL_MAP

def plot_confusion_matrix(all_y_true, all_y_pred, saving_dir, class_names):
    # 1. Compute the confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)

    # 2. Set up the matplotlib figure (300 DPI is standard for print)
    plt.figure(figsize=(18, 10), dpi=300)
    sns.set_theme(style="white") # Clean background

    # 3. Plot the heatmap
    # Use cmap='Blues' for a clean academic look, or 'viridis' for colorblind accessibility
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 12},  # Font size for numbers
                    cbar_kws={'label': 'Number of Samples'})

    # 4. Add formatting and labels
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, pad=15, fontweight='bold')

    # Rotate x-axis labels if your class names are long
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # 5. Save the figure in a high-quality format
    plt.tight_layout()
    plt.savefig(os.path.join(saving_dir, 'confusion_matrix_high_res.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(saving_dir, 'confusion_matrix_high_res.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"saved {os.path.join(saving_dir, 'confusion_matrix_high_res.pdf')}")
    print(f"saved {os.path.join(saving_dir, 'confusion_matrix_high_res.png')}")

def plot_classification_report_heatmap(all_y_true, all_y_pred, saving_dir, class_names):
    # 1. Get the classification report as a dictionary
    report_dict = classification_report(all_y_true, all_y_pred, 
                                        target_names=class_names, 
                                        output_dict=True)

    # 2. Convert to DataFrame and drop the 'support' column for the color mapping
    # (Support throws off the 0-1 color scale because it's a count, not a percentage)
    report_df = pd.DataFrame(report_dict).T
    metrics_df = report_df.drop(columns=['support'])
    
    # 3. Set up the figure
    # plt.figure(figsize=(18, 10), dpi=300)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, 10), dpi=300, gridspec_kw={"width_ratios": [3, 1]}
    )
    # 4. Plot heatmap
    
    sns.heatmap(metrics_df, annot=True, cmap='RdYlGn', fmt='.4f', 
                vmin=0.0, vmax=1.0, linewidths=1, linecolor='black',
                annot_kws={"size": 11}, ax=ax1)
    ax1.set_title("Classification Metrics per Class", fontsize=12, pad=10, fontweight='bold')
    # 5. Formatting
    # plt.title('Classification Metrics per Class', fontsize=14, pad=15, fontweight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)

    report_df.iloc[-3:, 3] = 0
    sns.heatmap(report_df[["support"]], annot=True, cmap="Reds", fmt='.0f', cbar=False, ax=ax2)# Clean up axes alignment
    ax2.set_title("Support per Class", fontsize=12, pad=10, fontweight='bold')
    ax2.set_yticks([])


    plt.tight_layout()
    plt.savefig(os.path.join(saving_dir, 'classification_report_heatmap.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(saving_dir, 'classification_report_heatmap.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"saved {os.path.join(saving_dir, 'classification_report_heatmap.pdf')}")
    print(f"saved {os.path.join(saving_dir, 'classification_report_heatmap.png')}")
    
def save_csv(all_img, all_y_true, all_y_pred, saving_dir):
    
    df = pd.DataFrame(
        {"image_path": all_img, "y_true": all_y_true, "y_pred": all_y_pred}
    )
    
    df.to_csv(os.path.join(saving_dir, 'test_results.csv'), index=False)
    
    print(f"saved {os.path.join(saving_dir, 'test_results.csv')}")
    
    
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
    
    transform_test = Compose(
        [
            LoadImage(image_only=True), 
            EnsureChannelFirst(), 
            ScaleIntensity(), 
            Resize(spatial_size=(224, 224), mode="bilinear")
            ]
            )

    print("loading NANFANGDataset...")
    test_csv = '/home/ma-user/work/rocket/Dataset/nanfang_capsule/preprocessed_oriname/v2_plus_v1sb_test.csv'
    image_root = '/home/ma-user/work/rocket/Dataset/nanfang_capsule/v2_plus_v1sb'
    test_dataset = NANFANGDataset(test_csv, image_root, transform_test)
    

    # Distributed sampler splits data across GPUs
    test_sampler = DistributedSampler(
        test_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    
    # Each GPU gets batch_size // world_size samples
    batch_size_per_gpu = 512 // world_size
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size_per_gpu, 
        sampler=test_sampler,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Model setup
    OUT_CHANNELS = 35
    ckpt = "/home/ma-user/work/rocket/Project/nanfang_capsule/checkpoints/exp_v2plusv1sb_resnet50_v2/best_model.pth"
    print(f"loading model weight from {ckpt}...")
    # model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=35).to(device)

    # resnet
    model = torchvision.models.resnet50()
    # model = torchvision.models.resnet101()
    # model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, OUT_CHANNELS)

    state_dict = torch.load(ckpt, weights_only=True)
    model_state_dict = state_dict.get("model_state_dict", "")
    model.load_state_dict(model_state_dict)
    
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    if rank == 0:
        start_time = time.time()

    test_sampler.set_epoch(1)
    
    if torch.npu.is_available():
        torch.npu.synchronize()
    if rank == 0:
        epoch_start = time.time()
    
    model.eval()
    y_true = []
    y_pred = []
    img_path = []
    correct = 0
    total = 0
    
    pbar = tqdm(
        test_loader, 
        desc="Testing", 
        disable=(rank != 0),
        dynamic_ncols=True
    )
    with torch.no_grad():
        for batch_idx, (data, target, img) in enumerate(pbar):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            pred = model(data).argmax(dim=1)
            
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            y_true.extend(target.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            img_path.extend(img)
        
    # Aggregate global metrics for correct/total
    metrics = torch.tensor([correct, total], dtype=torch.float32, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    global_correct, global_total = metrics.tolist()
    
    # FIX: Gather all predictions and targets to Rank 0 for the classification report
    y_true_tensor = torch.tensor(y_true, device=device)
    y_pred_tensor = torch.tensor(y_pred, device=device)
    # img_path_tensor = torch.tensor(list(img_path), device=device)
    img_path_tensor = list(img_path)
    
    # Create containers to receive gathered tensors
    gathered_y_true = [torch.zeros_like(y_true_tensor) for _ in range(world_size)]
    gathered_y_pred = [torch.zeros_like(y_pred_tensor) for _ in range(world_size)]
    gathered_img_lists = [None for _ in range(world_size)]
    
    dist.all_gather(gathered_y_true, y_true_tensor)
    dist.all_gather(gathered_y_pred, y_pred_tensor)
    dist.all_gather_object(gathered_img_lists, img_path_tensor)
    
    if torch.npu.is_available():
        torch.npu.synchronize()
        
    if rank == 0:
        # Concatenate tensors from all GPUs
        all_y_true = torch.cat(gathered_y_true).cpu().numpy()
        all_y_pred = torch.cat(gathered_y_pred).cpu().numpy()
        all_img = [
            path
            for rank_img_list in gathered_img_lists
            for path in rank_img_list
        ]

        
        # Note: DistributedSampler pads data to make it evenly divisible by world_size. 
        # If dataset size isn't perfectly divisible, you might have a few duplicate padded samples at the end.
        # You can slice it `[:len(test_dataset)]` to be perfectly accurate.
        all_y_true = all_y_true[:len(test_dataset)]
        all_y_pred = all_y_pred[:len(test_dataset)]
        all_img = all_img[:len(test_dataset)]

        print(f"\nGlobal Accuracy: {global_correct / global_total:.4f}")
        class_names = sorted(LABEL_MAP, key=LABEL_MAP.get)
        print(classification_report(all_y_true, all_y_pred, target_names=class_names, digits=4))
        
        total_time = time.time() - start_time
        print(f"Total Testing Time: {total_time:.2f}s")
        plot_confusion_matrix(all_y_true, all_y_pred, os.path.dirname(ckpt), class_names)
        plot_classification_report_heatmap(all_y_true, all_y_pred, os.path.dirname(ckpt), class_names)
        
        save_csv(all_img, all_y_true, all_y_pred, os.path.dirname(ckpt))
    dist.destroy_process_group()

if __name__ == "__main__":
    # torchrun sets these environment variables automatically
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    setup(local_rank, world_size)
