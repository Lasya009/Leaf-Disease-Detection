import tempfile
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from src.data import make_dataloaders
from src.model import build_model
from src.train import evaluate
from src.infer import predict


def create_synthetic_dataset(root_dir, num_classes=3, samples_per_class=5, img_size=224):
    root_dir = Path(root_dir)
    classes = [f'class_{i}' for i in range(num_classes)]

    for split in ['train', 'val']:
        for cls in classes:
            (root_dir / split / cls).mkdir(parents=True, exist_ok=True)

    with open(root_dir / 'classes.txt', 'w') as f:
        for c in classes:
            f.write(f'{c}\n')

    for split in ['train', 'val']:
        for cls_idx, cls in enumerate(classes):
            for i in range(samples_per_class):
                img_array = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img_path = root_dir / split / cls / f'img_{i}.jpg'
                img.save(img_path)

    return root_dir, classes


def test_e2e():
    print('\n=== E2E Test: Synthetic Dataset ===\n')

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print('1. Creating synthetic dataset...')
        data_dir, classes = create_synthetic_dataset(tmpdir / 'data', num_classes=3, samples_per_class=5)
        print(f'   Classes: {classes}')

        print('2. Loading dataloaders...')
        train_loader, val_loader, loaded_classes = make_dataloaders(data_dir, batch_size=2, img_size=224, num_workers=0)
        print(f'   Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}')
        assert loaded_classes == classes

        print('3. Building custom CNN model...')
        model = build_model(num_classes=len(classes), model_name='custom_cnn', pretrained=False)
        device = torch.device('cpu')
        model = model.to(device)
        print(f'   Model: {model.__class__.__name__}')

        print('4. Running forward pass on a batch...')
        x, y = next(iter(train_loader))
        x = x.to(device)
        out = model(x)
        assert out.shape == (x.shape[0], len(classes))
        print(f'   Output shape: {out.shape} (expected {(x.shape[0], len(classes))})')

        print('5. Evaluating on validation set...')
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f'   Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')

        print('6. Training for 1 epoch...')
        import torch.nn as nn
        import torch.optim as optim
        from tqdm import tqdm

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        print(f'   Train loss: {train_loss:.4f}')

        print('7. Saving checkpoint...')
        ckpt_dir = tmpdir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / 'test.pth'
        torch.save({'model_state_dict': model.state_dict(), 'classes': classes, 'model_name': 'custom_cnn', 'img_size': 224}, ckpt_path)
        print(f'   Checkpoint saved: {ckpt_path}')

        print('8. Running inference on a test image...')
        test_img_path = tmpdir / 'test_img.jpg'
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(test_img_path)

        results = predict(str(ckpt_path), str(test_img_path), topk=2)
        print(f'   Top-2 predictions: {results}')
        assert len(results) == 2
        assert all(cls in classes for cls, prob in results)

        print('\n✓ E2E test PASSED\n')


if __name__ == '__main__':
    test_e2e()
