# 🚀 PyTorch-ML

> **A hands-on, beginner-friendly repository for learning PyTorch fundamentals and building neural networks from scratch.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.0-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org)

---

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Learning Path](#-learning-path)
- [Notebooks Overview](#-notebooks-overview)
- [Visual Resources](#-visual-resources)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 About

**PyTorch-ML** is a curated collection of educational resources designed to help you master PyTorch for machine learning and deep learning. Whether you're a student, researcher, or practitioner, this repo provides:

✅ **Clear, modular explanations** of PyTorch concepts  
✅ **Interactive Jupyter notebooks** with runnable code  
✅ **Visual diagrams** to demystify complex topics like CNNs  
✅ **Production-ready patterns** for data pipelines and model training  


---

## ✨ Features

- 🔹 **Dynamic Computation Graphs**: Learn why PyTorch's define-by-run approach is ideal for research and debugging
- 🔹 **End-to-End ML Pipeline**: From data ingestion → preprocessing → modeling → evaluation → deployment
- 🔹 **Tensor Mastery**: Shape handling, data types, indexing, and GPU acceleration
- 🔹 **Neural Network Fundamentals**: Linear models, activation functions (ReLU), and building complexity
- 🔹 **CNN Deep Dive**: Convolutions, pooling, feature maps with animated visualizations
- 🔹 **Data Pipelines**: Efficient `Dataset` and `DataLoader` implementations for scalable training

---

## 📁 Repository Structure

```
PyTorch-ML/
├── PyTorch_Fundamentals/
│   ├── Module 1.md          # Why PyTorch? ML Pipeline overview
│   ├── Module 2.md          # Building neural networks, tensors, normalization
│   ├── Module 3.md          # Production-grade data handling
│   ├── Module 4.md          # CNN explanations & architectures
│   └── Notebooks/
│       ├── SimpleNN.ipynb           # Basic feedforward network
│       ├── Mnist_Digits_classifier.ipynb  # CNN on MNIST dataset
│       ├── EnglishLetters_Classifier.ipynb # Custom image classification
│       └── DataPipelines.ipynb      # Custom Dataset & DataLoader patterns
│
├── .gitignore
└── README.md
```

---

## 📚 Learning Path

Follow the modules in order for a structured learning experience:

| Module | Topic | Key Concepts |
|--------|-------|-------------|
| **Module 1** | Why PyTorch? | Dynamic graphs, research advantages, ML pipeline overview |
| **Module 2** | Neural Networks 101 | Tensors, linear models, ReLU, normalization, training loop |
| **Module 3** | Data Engineering | `Dataset`, `DataLoader`, transforms, batching, preprocessing |
| **Module 4** | Convolutional Networks | Convolutions, pooling, feature maps, CNN architecture design |

---

## 📓 Notebooks Overview

| Notebook | Description | Dataset | Skills Practiced |
|----------|-------------|---------|-----------------|
| `SimpleNN.ipynb` | Build a basic regression model | Synthetic delivery-time data | `nn.Linear`, `MSELoss`, SGD optimizer |
| `Mnist_Digits_classifier.ipynb` | Classify handwritten digits | MNIST (via `torchvision`) | CNN layers, `CrossEntropyLoss`, evaluation metrics |
| `EnglishLetters_Classifier.ipynb` | Custom letter recognition | EMNIST-style data | Data augmentation, custom training loops |
| `DataPipelines.ipynb` | Efficient data loading patterns | Any CSV/image dataset | `torch.utils.data.Dataset`, transforms, multiprocessing |

---

## ⚙️ Prerequisites

- Basic Python programming (functions, classes, NumPy)
- Fundamental ML concepts (loss functions, gradient descent, train/test split)
- Optional but helpful: Linear algebra (matrix operations) and calculus (derivatives)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-idea`
3. Commit changes: `git commit -m 'Add: your feature'`
4. Push to the branch: `git push origin feat/your-idea`
5. Open a Pull Request

✅ **Good contributions include**:
- Bug fixes in notebooks
- New tutorial notebooks (with clear explanations)
- Improved visualizations or diagrams
- Documentation enhancements

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

> 💬 **Have questions or suggestions?**  
> Open an [issue](https://github.com/GeekyAbs/PyTorch-ML/issues) or reach out to [@GeekyAbs](https://github.com/GeekyAbs) on GitHub!

⭐ **If you found this repo helpful, please star it!** It helps others discover these resources.