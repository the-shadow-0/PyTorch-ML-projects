import argparse
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer

def load_mnist_model(path: str, device: torch.device) -> nn.Module:
    class MNISTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 5 * 5, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            x = self.fc(x)
            return x

    model = MNISTModel().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_imdb_model(path: str, device: torch.device) -> (nn.Module, BertTokenizer):
    class IMDBModel(nn.Module):
        def __init__(self, vocab_size: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 128)
            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embedding(x)
            x = x.mean(dim=1)
            x = self.fc(x)
            return x

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    model = IMDBModel(vocab_size).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer

def load_california_model(path: str, device: torch.device) -> nn.Module:
    class CaliforniaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layers(x)

    model = CaliforniaModel().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def run_mnist(model: nn.Module, index: int, device: torch.device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(root=".", train=False, download=True, transform=transform)
    if not (0 <= index < len(test_dataset)):
        print(f"Error: index {index} out of bounds (must be 0–{len(test_dataset)-1}).")
        return
    image, label = test_dataset[index]
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
    print(f"MNIST → Prediction: {pred}, Ground Truth: {label}")

def run_imdb(model: nn.Module, tokenizer: BertTokenizer, text: str, device: torch.device):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    )
    input_ids = tokens["input_ids"].to(device)
    with torch.no_grad():
        output = model(input_ids)
        score = output.squeeze().item()
        sentiment = "Positive" if score > 0.5 else "Negative"
    print(f"IMDB → Review: \"{text}\"")
    print(f"       Prediction: {sentiment} (score={score:.4f})")

def run_california(model: nn.Module, index: int, device: torch.device):
    data = fetch_california_housing()
    X = StandardScaler().fit_transform(data.data)
    if not (0 <= index < X.shape[0]):
        print(f"Error: index {index} out of bounds (must be 0–{X.shape[0]-1}).")
        return
    sample = torch.tensor(X[index], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(sample).item()
    actual = data.target[index]
    print(f"California Housing → Prediction: ${output * 100000:.2f}")
    print(f"                     Ground Truth: ${actual * 100000:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a pre-trained PyTorch model (MNIST, IMDB, or California Housing)."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["mnist", "imdb", "california"],
        help="Which type of model to run: 'mnist', 'imdb', or 'california'."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the saved .pth file (e.g. models/mnist_cnn.pth)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "For 'mnist' or 'california': an integer index into the test set.\n"
            "For 'imdb': a text review in quotes (e.g. \"A great movie!\")."
        )
    )
    args = parser.parse_args()
    if not os.path.isfile(args.model):
        print(f"Error: Model file not found at '{args.model}'.")
        print("       Please ensure you trained & saved it with exactly this name.")
        exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.task == "mnist":
        try:
            idx = int(args.input)
        except ValueError:
            print("Error: --input for 'mnist' must be an integer index (e.g. 0).")
            exit(1)
        model = load_mnist_model(args.model, device)
        run_mnist(model, idx, device)
    elif args.task == "imdb":
        model, tokenizer = load_imdb_model(args.model, device)
        run_imdb(model, tokenizer, args.input, device)
    elif args.task == "california":
        try:
            idx = int(args.input)
        except ValueError:
            print("Error: --input for 'california' must be an integer index (e.g. 5).")
            exit(1)
        model = load_california_model(args.model, device)
        run_california(model, idx, device)

