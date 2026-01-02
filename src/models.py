import torch
import torch.nn as nn

class Denoiser1DCNN(nn.Module):
    def __init__(self):
        super(Denoiser1DCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
# Example usage:
# model = Denoiser1DCNN()               
# noisy_signal = torch.randn(1, 1, 1024)  # Batch size 1, 1 channel, length 1024
# denoised_signal = model(noisy_signal)
# print(denoised_signal.shape)  # Should be [1, 1, 1024]
