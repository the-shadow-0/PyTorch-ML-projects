# PyTorch ML Projects

Welcome to **PyTorch-ML-projects**! This is a beginner-friendly collection of machine learning projects built using PyTorch.
Each example demonstrates a real-world task:

* 🧠 Digit recognition (MNIST)
* 💬 Sentiment analysis (IMDB movie reviews)
* 🏡 Housing price prediction (California housing dataset)

The goal is to test my knowledge and maybe inspire beginners how to build, train, and use PyTorch models for different kinds of problems: classification, NLP, and regression.

---

## What’s Inside

### `mnist_classifier.py`

Train a CNN to recognize handwritten digits (0–9).

* Loads MNIST data (28x28 grayscale images)
* Simple CNN with convolution + fully connected layers
* Trains for 5 epochs
* Saves model to `models/mnist_cnn.pth`

**To run:**

```bash
python3 mnist_classifier.py
```

### `imdb_sentiment.py`

Train a text classifier to predict movie review sentiment.

* Uses HuggingFace `datasets` to load IMDB
* Tokenizes reviews with BERT tokenizer
* Small neural network: embedding + mean pooling + FC layers
* Saves model to `models/imdb_sentiment_hf.pth`

**To run:**

```bash
pip install datasets transformers tqdm
python3 imdb_sentiment.py
```

### `california_regression.py`

Train a regressor to predict housing prices in California.

* Loads dataset with scikit-learn
* Neural net: 8 inputs → 64 hidden → 1 output
* Trains for 10 epochs, prints MSE loss
* Saves model to `models/california_model.pth`

**To run:**

```bash
pip install scikit-learn
python3 california_regression.py
```

### `inference.py`

Run predictions on any trained model without re-training.

Supports:

* `mnist`: test by index (0 = first test image)
* `imdb`: input a text string (e.g., "This movie was great!")
* `california`: test housing prediction by index

All use GPU if available.

**Examples:**

```bash
python3 inference.py --task mnist --model models/mnist_cnn.pth --input 0
python3 inference.py --task imdb --model models/imdb_sentiment_hf.pth --input "Awesome movie, I loved it."
python3 inference.py --task california --model models/california_model.pth --input 5
```

---

## Getting Started

1. Clone the repo:

```bash
git clone https://github.com/the-shadow-0/PyTorch-ML-projects.git
cd PyTorch-ML-projects
```

2. Install dependencies:

```bash
pip install torch torchvision datasets transformers tqdm scikit-learn matplotlib
```

3. Run any training script (see above), or test models with `inference.py`

4. Your trained models will be saved to the `models/` folder if you prefere to re-train any model by yourself.

---

## Project Structure

```
PyTorch-ML-projects/
├── mnist_classifier.py          # Digit classification
├── imdb_sentiment.py           # Movie review sentiment analysis
├── california_regression.py    # Housing price prediction
├── inference.py                # Quick predictions
├── models/                     # Saved trained models
│   ├── mnist_cnn.pth
│   ├── imdb_sentiment_hf.pth
│   └── california_model.pth
└── README.md                   # You're reading this :)
```

---

## Notes

* GPU is used automatically if available (PyTorch's `cuda` support)
* Models save to `models/` directory by default
* All datasets are downloaded automatically on first run

---

Made with ❤️ using PyTorch.

Feel free to fork, improve, or try on your own data!
