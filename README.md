# Fake News Classifier using LSTM

A deep learning-based fake news classification project using **Long Short-Term Memory (LSTM)** networks. The model analyzes the textual content of news articles and classifies them as **Fake** or **True**.

## Overview

Fake news can spread rapidly through online platforms, making automated detection an important Natural Language Processing (NLP) task.

In this project, an LSTM-based neural network was developed to classify news articles based on their textual content. The dataset consists of separate files containing fake and true news articles, which were combined and assigned binary labels before training.

The trained model achieved approximately **93% classification accuracy**.

## Dataset

The dataset contains two separate CSV files:

* `Fake.csv` — contains fake news articles
* `True.csv` — contains true news articles

Each dataset contains information such as:

* `title`
* `text`
* `subject`
* `date`

Since the original files did not contain an explicit target column, binary labels were assigned:

* `0` → Fake News
* `1` → True News

The combined dataset contains **44,898 articles**.

## Approach

The project follows a typical NLP text-classification pipeline:

1. Load the fake and true news datasets.
2. Assign binary labels to the two datasets.
3. Combine the datasets into a single dataset.
4. Preprocess the news text.
5. Tokenize the text and convert words into numerical sequences.
6. Apply padding so that the input sequences have a fixed length.
7. Split the dataset into training and testing sets.
8. Convert the text sequences into dense vector representations using an embedding layer.
9. Pass the sequences through an LSTM layer to learn sequential patterns in the text.
10. Use a sigmoid output layer for binary classification.
11. Train and evaluate the model.

## Model Architecture

The model consists of three main components:

```text
Input Text
    ↓
Tokenization & Padding
    ↓
Embedding Layer
    ↓
LSTM Layer (100 units)
    ↓
Dense Layer (Sigmoid)
    ↓
Fake / True
```

### Embedding Layer

The embedding layer converts integer word indices into dense vector representations.

The embedding dimension used in the model is **40**.

### LSTM Layer

An LSTM layer with **100 units** is used to capture sequential relationships and contextual information in the news text.

### Output Layer

A single neuron with a **sigmoid activation function** produces the probability for binary classification.

## Training

The model was compiled using:

* **Loss:** Binary Cross-Entropy
* **Optimizer:** Adam
* **Metric:** Accuracy
* **Epochs:** 10
* **Batch Size:** 64

## Results

The model achieved approximately:

**93% accuracy**

This demonstrates that the LSTM model was able to learn useful patterns from the textual content for distinguishing between fake and true news articles.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Natural Language Processing
* LSTM
* Jupyter Notebook

## Repository Structure

```text
FakeNewsClassifierLSTM/
│
├── FakeNewsClassifierUsingLSTM.ipynb
├── README.md
└── requirements.txt
```

> The dataset files are not included in the repository because they are large and are not required to understand the implementation.


## Author

**Arshia Sood**

GitHub: [Arshia-Sood](https://github.com/Arshia-Sood)
