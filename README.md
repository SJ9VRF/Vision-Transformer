# Vision Transformer

This repository contains the implementation of a Vision Transformer (ViT) model trained and evaluated on the CIFAR-10 dataset. The project is structured to demonstrate best practices in organizing deep learning code using PyTorch.


![Screenshot_2024-08-08_at_9 45 42_PM-removebg-preview](https://github.com/user-attachments/assets/c46a5d7b-8b45-4436-9e0a-03d0a093cdcc)

## Project Structure

- `models/`: Contains the Vision Transformer model definition. This module is the core of the model's architecture, defining how images are processed into embeddings and handled through the transformer layers.

- `data/`: Manages data loading and preprocessing. It includes functionality to download the CIFAR-10 dataset, apply transformations, and prepare it for training and validation.

- `utils/`: Houses additional utility functions, such as data transformations that are essential for image preprocessing before they are fed into the model.

- `training/`: Contains the training and evaluation logic. This includes defining the loss functions, the optimizer, and the training loop that also handles model evaluation on the validation set.

- `configs/`: Stores configuration files which dictate model parameters, training settings, and other configurations that allow for easy adjustments without modifying the codebase.

- `main.py`: The main script used to run the entire training and evaluation process. It ties together all the modules and allows for the execution of the model training.

- `requirements.txt`: Lists all the necessary Python dependencies required to run the project. This ensures any environment setup can be replicated to support the model's functionality.

## Installation

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/vision_transformer.git
cd vision_transformer
pip install -r requirements.txt
```

