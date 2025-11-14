#!/usr/bin/env python
"""Quick demo: train for 1 batch, save checkpoint, and run inference."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.data import make_dataloaders
from src.model import build_model
import torch.nn as nn
import torch.optim as optim

def demo():
    device = torch.device('cpu')
    print('Loading data...')
    train_loader, val_loader, classes = make_dataloaders(
        './data/processed', batch_size=8, img_size=224, num_workers=0
    )
    num_classes = len(classes)
    print(f'Classes: {num_classes}, Sample classes: {classes[:3]}')

    print('Building model...')
    model = build_model(num_classes=num_classes, model_name='custom_cnn', pretrained=False)
    model = model.to(device)

    print('Training for 1 batch...')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Batch loss: {loss.item():.4f}')
        break  # just 1 batch

    print('Saving checkpoint...')
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': classes,
        'model_name': 'custom_cnn',
        'img_size': 224,
    }, './checkpoints/best.pth')
    print('✓ Checkpoint saved to ./checkpoints/best.pth')

    print('Running inference on first validation image...')
    from src.infer import predict
    val_dir = './data/processed/val'
    for cls in os.listdir(val_dir):
        cls_dir = os.path.join(val_dir, cls)
        if os.path.isdir(cls_dir):
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cls_dir, img_name)
                    results = predict('./checkpoints/best.pth', img_path, topk=3)
                    print(f'\nImage: {img_name}')
                    print('Predictions:')
                    for pred_cls, prob in results:
                        print(f'  {pred_cls}: {prob:.4f}')
                    return

if __name__ == '__main__':
    demo()
