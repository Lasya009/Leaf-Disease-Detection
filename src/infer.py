"""
Single-image inference script.

Example:
python .\src\infer.py --checkpoint .\checkpoints\best.pth --image .\samples\leaf.jpg

"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from src.model import build_model
from src.data import get_transforms


def load_checkpoint(path: Path):
    data = torch.load(path, map_location='cpu')
    return data


def predict(checkpoint_path, image_path, topk=3, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    ckpt = load_checkpoint(Path(checkpoint_path))
    classes = ckpt.get('classes')
    model_name = ckpt.get('model_name', 'custom_cnn')
    img_size = ckpt.get('img_size', 224)

    model = build_model(num_classes=len(classes), model_name=model_name, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    img = Image.open(image_path).convert('RGB')
    transform = get_transforms(img_size=img_size, train=False)
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).squeeze(0)
        topk_vals, topk_idx = torch.topk(probs, k=min(topk, len(classes)))

    results = [(classes[i], float(topk_vals[j])) for j, i in enumerate(topk_idx.tolist())]
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--topk', type=int, default=3)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    res = predict(args.checkpoint, args.image, topk=args.topk)
    print('Predictions:')
    for cls, prob in res:
        print(f"{cls}: {prob:.4f}")
