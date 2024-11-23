import numpy as np
import torch
import torch.nn as nn
import pandas as pd

class Generator(nn.Module):
    def __init__(self, num_classes, latent_dim=100):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, z):
        return self.model(z)

class DataGenerator:
    def __init__(self, num_classes=3, latent_dim=100):
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.generator = Generator(num_classes, latent_dim)
        
    def generate_dataset(self, 
                        num_samples=1000,
                        noise_level=0.1,
                        bias_strength=0.3,
                        class_imbalance=None):
        """
        Generate synthetic prediction data
        
        Args:
            num_samples: Number of samples to generate
            noise_level: Amount of noise to add to predictions (0-1)
            bias_strength: Strength of systematic bias in predictions (0-1) 
            class_imbalance: List of class probabilities (optional)
        
        Returns:
            DataFrame with predictions and ground truth
        """
        
        # Generate latent vectors
        z = torch.randn(num_samples, self.latent_dim)
        
        # Generate base predictions
        with torch.no_grad():
            predictions = self.generator(z).numpy()
        
        # Add noise
        noise = np.random.normal(0, noise_level, predictions.shape)
        predictions = predictions + noise
        predictions = np.clip(predictions, 0, 1)
        
        # Normalize to sum to 1
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
        
        # Add systematic bias
        if bias_strength > 0:
            bias = np.random.random(self.num_classes)
            bias = bias / bias.sum()
            bias = (bias * bias_strength) + (1 - bias_strength)
            predictions = predictions * bias
            predictions = predictions / predictions.sum(axis=1, keepdims=True)
        
        # Generate ground truth
        if class_imbalance is None:
            class_imbalance = [1/self.num_classes] * self.num_classes
        ground_truth = np.random.choice(
            range(1, self.num_classes + 1),
            size=num_samples,
            p=class_imbalance
        )
        
        # Combine predictions and ground truth
        data = np.column_stack([predictions, ground_truth])
        
        return pd.DataFrame(data)

    def save_to_csv(self, df, filename):
        """Save generated data to CSV file"""
        df.to_csv(filename, index=False, header=False)
