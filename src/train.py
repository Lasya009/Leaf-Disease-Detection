"""
Training script for leaf disease detection.

Example (PowerShell):
python .\src\train.py --data-dir .\data\processed --epochs 10 --batch-size 32 --output-dir .\checkpoints --model-name custom_cnn

"""
import argparse
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data import make_dataloaders
from src.model import build_model


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return running_loss / total, correct / total


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_loader, val_loader, classes = make_dataloaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)
    num_classes = len(classes)

    model = build_model(num_classes=num_classes, model_name=args.model_name, pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Setup scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, patience=2)

    best_val_acc = 0.0
    start_epoch = 1
    os.makedirs(args.output_dir, exist_ok=True)

    # Resume from checkpoint
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f'Resuming from {ckpt_path}')
            data = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(data['model_state_dict'])
            if 'optimizer_state_dict' in data:
                optimizer.load_state_dict(data['optimizer_state_dict'])
            if 'scheduler_state_dict' in data and scheduler is not None:
                scheduler.load_state_dict(data['scheduler_state_dict'])
            start_epoch = data.get('epoch', 1) + 1
            best_val_acc = data.get('best_val_acc', 0.0)
            print(f'Resumed: best_val_acc={best_val_acc:.4f}, starting from epoch {start_epoch}')
        else:
            print(f'Resume path not found: {ckpt_path}, starting from scratch')

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Step scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'classes': classes,
            'model_name': args.model_name,
            'img_size': args.img_size,
            'best_val_acc': best_val_acc,
        }
        if scheduler is not None:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()

        last_path = Path(args.output_dir) / f'last_epoch_{epoch}.pth'
        torch.save(ckpt, last_path)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = Path(args.output_dir) / 'best.pth'
            torch.save(ckpt, best_path)
            print(f'Saved best checkpoint to {best_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model-name', type=str, default='resnet18', help='custom_cnn | resnet18 | resnet50')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--scheduler', type=str, default=None, help='step | reduce_on_plateau')
    parser.add_argument('--step-size', type=int, default=5, help='LR scheduler step size (for step scheduler)')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR decay factor')
    return parser.parse_args()
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
