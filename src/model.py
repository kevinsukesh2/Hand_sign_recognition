"""PyTorch model definition for hand sign classification."""

import torch.nn as nn


class GestureMLP(nn.Module):
    """Simple multilayer perceptron for landmark-based gesture classification."""

    def __init__(self, input_size=126, num_classes=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """Run a forward pass through the classifier."""
        return self.network(x)


def main():
    """Show a short message for direct script execution."""
    print("Phase 5 defines the GestureMLP classifier.")


if __name__ == "__main__":
    main()
