import torch

from src.model import build_model


def test_custom_cnn_forward():
    model = build_model(num_classes=5, model_name='custom_cnn', pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 5)


if __name__ == '__main__':
    test_custom_cnn_forward()
    print('Smoke test passed')
