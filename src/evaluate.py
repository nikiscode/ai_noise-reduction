import torch
import numpy as np
import matplotlib.pyplot as plt
from models import Denoiser1DCNN
import os

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Denoiser1DCNN().to(device)
model_path = "models/denoiser_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model not found. Train the model first.")

model.load_state_dict(torch.load(model_path))
model.eval()

# Load a noisy signal
noisy_files = sorted([f for f in os.listdir("data/noisy_signals") if f.endswith(".npy")])
if len(noisy_files) == 0:
    raise FileNotFoundError("No noisy signals found. Run train.py first.")

noisy_signal = np.load(os.path.join("data/noisy_signals", noisy_files[0]))
noisy_tensor = torch.tensor(noisy_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# Denoise
with torch.no_grad():
    denoised = model(noisy_tensor).cpu().numpy().squeeze()

# Plot
plt.figure(figsize=(10,4))
plt.plot(noisy_signal, label="Noisy Signal")
plt.plot(denoised, label="Denoised Signal")
plt.title("AI-Based Optical Signal Denoising")
plt.legend()
plt.show()
