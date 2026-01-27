# Basic Decoder Transformer Language Model

This repository presents a **from-scratch implementation of a GPT-style decoder-only Transformer** using PyTorch. The project is intended for **educational and research-oriented purposes**, focusing on architectural correctness, training mechanics, and clarity of implementation rather than output quality.

---

## Abstract

We implement a character-level autoregressive language model based on the Transformer decoder architecture. The model employs causal self-attention, residual connections, and layer normalization, trained using maximum likelihood estimation to predict the next token given a context window. This implementation serves as a minimal and transparent reference for understanding GPT-style models.

---

## Model Architecture

The architecture follows the standard decoder-only Transformer design:

- Token Embedding Layer
- Positional Embedding Layer
- Stack of Transformer Blocks, each containing:
  - Multi-Head Causal Self-Attention
  - Feed-Forward Network
  - Residual Connections
  - Layer Normalization
- Final Linear Projection to Vocabulary Space

The model operates at the character level, using a simple character-wise tokenization scheme. Each unique character in the dataset is mapped to an integer index, and the model is trained to autoregressively predict the next character given a fixed-length context window.

The implementation follows standard Transformer decoder principles, informed by the original [Transformer paper](https://arxiv.org/abs/1706.03762) (*Attention Is All You Need*, Vaswani et al., 2017), with additional guidance from [Andrej Karpathy's lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=8).

---

## Training Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Context Length (Block Size) | 256 |
| Embedding Dimension | 384 |
| Number of Attention Heads | 6 |
| Number of Transformer Layers | 6 |
| Dropout | 0.2 |
| Learning Rate | 3.00 × 10⁻⁴ |
| Optimizer | AdamW |
| Learning Rate Scheduler | ReduceLROnPlateau |
| Gradient Clipping | 1.0 |
| Precision | Automatic Mixed Precision (AMP) |
| Training Iterations | 5000 |

---

## Hardware & Training Time

- **GPU:** NVIDIA RTX 4060
- **Total Training Time:** 22.43 minutes

### Final Metrics

- **Training Loss:** 0.679883
- **Validation Loss:** 1.134118

These metrics are reported for completeness; performance optimization was not a primary objective.

---

## Dataset

The training data consists of a single text file composed of movie transcript excerpts from:

- *Interstellar*
- *Prisoners*
- *The Revenant*
- *Oppenheimer*
- *Nightcrawler*
- *Zodiac*
- *Memento*
- *Se7en*
- *The Prestige*

The dataset is intentionally simple and serves only to provide sequential text for learning character-level distributions.

---

## Sample Generation (Excerpt)
```
Cooper, showly, something or many young sin.

LOKI: Hey, have you, Dad.

Let's go!

Don't do open books!

Get up!

Get out!

Who put I thought me on. It's…

We believe you got that way we've never seen them.

I've got some and kids for long gonna life them.
```

*Text quality is not emphasized; the sample demonstrates successful autoregressive generation.*

---

## Repository Structure
```
basic-lm/
├── checkpoints/
│   └── best_model.pt
├── dataset.txt
├── gpt.py
├── gpt_train.py
├── generate_sample.py
├── generated_sample.txt
├── training_metrics.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Scope and Limitations

This project prioritizes:

- Conceptual correctness
- Training loop transparency
- Architectural clarity

It does **not** aim to:

- Compete with large-scale pretrained models
- Optimize generation quality
- Use large or curated datasets

---

## License

MIT License