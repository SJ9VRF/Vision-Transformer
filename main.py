# The main execution script that uses other modules.
from models.vision_transformer import VisionTransformer
from data.data_loader import get_loaders
from training.trainer import train, evaluate
import json

def main():
    with open('configs/config.json') as f:
        config = json.load(f)

    model = VisionTransformer(config)
    train_loader, val_loader, test_loader = get_loaders(config)
    train(model, train_loader, config)
    evaluate(model, val_loader)

if __name__ == "__main__":
    main()
