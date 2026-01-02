import torch
from torch.utils.data import DataLoader
from dataset import OpticalSignalDataset
from models import Denoiser1DCNN
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

# Paths
clean_dir = "data/raw_signals"
noisy_dir = "data/noisy_signals"
os.makedirs("models", exist_ok=True)

# Generate noisy signals if not exists
if len(os.listdir(noisy_dir)) == 0:
    os.makedirs(noisy_dir, exist_ok=True)
    for f in os.listdir(clean_dir):
        signal = np.load(os.path.join(clean_dir, f))
        noise = np.random.normal(0, 0.1, size=signal.shape)
        noisy_signal = signal + noise
        np.save(os.path.join(noisy_dir, f), noisy_signal)
    print("Noisy signals generated.")

# Dataset & DataLoader
dataset = OpticalSignalDataset(clean_dir, noisy_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Denoiser1DCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 50
for epoch in range(epochs):
    running_loss = 0.0
    for noisy, clean in dataloader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.6f}")

# Save Model
torch.save(model.state_dict(), "models/denoiser_model.pth")
print("Training complete. Model saved at models/denoiser_model.pth")
