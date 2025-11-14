# Leaf disease detection — PlantVillage dataset

This repository provides a complete leaf disease detection pipeline using the PlantVillage dataset (Kaggle).
It includes a custom CNN trained from scratch and transfer-learning models (ResNet) with training, checkpointing, and inference.

## Dataset
- Source: https://www.kaggle.com/datasets/emmarex/plantdisease
- ~50,000 images of healthy and disease-affected leaves from various plant species
- Organized by disease type and plant name (e.g., `Apple___Apple_scab`, `Corn___Healthy`)

## Project structure
```
.
├── src/
│   ├── data.py           # Dataset loading, augmentations, transforms
│   ├── model.py          # CustomCNN and transfer-learning models
│   ├── train.py          # Training loop with resume, scheduler, checkpointing
│   ├── infer.py          # Single-image inference
│   └── __init__.py
├── scripts/
│   └── prepare_data.py   # Prepare PlantVillage into train/val/test splits
├── tests/
│   ├── test_model_smoke.py    # Quick smoke test (forward pass only)
│   └── test_e2e.py            # End-to-end integration test (synthetic data)
├── configs/
│   └── config.yaml       # Default hyperparameters
├── requirements.txt      # Python dependencies
└── README.md
```

## Quick setup (Windows PowerShell)

### 1. Create and activate virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
python -m pip install -r requirements.txt
```

### 3. Download & prepare dataset

**Option A: Kaggle CLI (recommended)**
```powershell
# Install: https://github.com/Kaggle/kaggle-api
# Configure credentials (place kaggle.json in ~/.kaggle/)
kaggle datasets download -d emmarex/plantdisease -p .\data --unzip
```

**Option B: Manual download**
- Download from https://www.kaggle.com/datasets/emmarex/plantdisease
- Unzip into `data/` folder

### 4. Prepare data (creates train/val/test splits)
```powershell
python .\scripts\prepare_data.py --data-dir .\data --out-dir .\data\processed --val-split 0.15 --test-split 0.1
```

## Training

### Quick tests (no dataset needed)

**Smoke test (forward pass only)**
```powershell
python .\tests\test_model_smoke.py
```

**End-to-end test (creates synthetic data)**
```powershell
python .\tests\test_e2e.py
```

### Train custom CNN (from scratch, CPU-friendly)
```powershell
python .\src\train.py `
  --data-dir .\data\processed `
  --epochs 10 `
  --batch-size 16 `
  --model-name custom_cnn `
  --output-dir .\checkpoints
```

### Train ResNet18 (transfer learning, recommended)
```powershell
python .\src\train.py `
  --data-dir .\data\processed `
  --epochs 10 `
  --batch-size 32 `
  --model-name resnet18 `
  --pretrained `
  --output-dir .\checkpoints
```

### Train with learning rate scheduler
```powershell
python .\src\train.py `
  --data-dir .\data\processed `
  --epochs 20 `
  --batch-size 32 `
  --model-name resnet18 `
  --pretrained `
  --scheduler step `
  --step-size 5 `
  --gamma 0.1 `
  --output-dir .\checkpoints
```

### Resume training from checkpoint
```powershell
python .\src\train.py `
  --data-dir .\data\processed `
  --epochs 20 `
  --resume .\checkpoints\last_epoch_5.pth `
  --output-dir .\checkpoints
```

## Inference

### Single-image prediction
```powershell
python .\src\infer.py `
  --checkpoint .\checkpoints\best.pth `
  --image .\samples\leaf.jpg `
  --topk 3
```

Example output:
```
Predictions:
Apple___Apple_scab: 0.9234
Apple___Healthy: 0.0652
Apple___Black_rot: 0.0114
```

## Model comparison

| Model | Training time | Accuracy | Best for |
|-------|---|---|---|
| **custom_cnn** | Fast (CPU) | Baseline | Prototyping, CPU constraints |
| **resnet18** (pretrained) | Medium (GPU) | High | Good balance, most use cases |
| **resnet50** (pretrained) | Slow (GPU) | Very high | Production, GPU available |

## Training arguments

```
--data-dir         (required) Path to processed data (train/val folders)
--output-dir       Checkpoint directory (default: checkpoints)
--epochs           Number of epochs (default: 10)
--batch-size       Batch size (default: 32)
--lr               Learning rate (default: 1e-3)
--model-name       custom_cnn | resnet18 | resnet50 (default: resnet18)
--img-size         Image size (default: 224)
--num-workers      DataLoader workers (default: 4)
--pretrained       Use ImageNet pretraining (for ResNet)
--resume           Checkpoint to resume from
--scheduler        LR scheduler: step | reduce_on_plateau
--step-size        Scheduler step size (default: 5)
--gamma            LR decay factor (default: 0.1)
```

## Checkpoints

- **best.pth**: Best validation accuracy (resumable)
- **last_epoch_N.pth**: Latest epoch checkpoint (resumable)

Contents:
```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),  # if used
    'epoch': epoch,
    'best_val_acc': best_val_acc,
    'classes': class_list,
    'model_name': 'custom_cnn',
    'img_size': 224,
}
```

## Data augmentations

**Training**: RandomResizedCrop, HFlip, VFlip, Rotation(±15°), ColorJitter
**Val/Test**: Resize + CenterCrop (no augmentation)

## Key features

✓ Custom CNN (train from scratch)
✓ Transfer learning (ResNet18/50)
✓ Resume training from checkpoint
✓ Learning rate scheduling
✓ Data augmentation
✓ Per-epoch checkpointing
✓ Single-image inference
✓ GPU/CPU support
✓ Windows PowerShell ready

## Notes

- GPU auto-detected; uses CPU fallback
- ImageNet normalization (standard for transfer learning)
- Default splits: 75% train, 15% val, 10% test
- Use `custom_cnn` with batch size 8–16 for CPU-only