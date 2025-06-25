import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # Simple CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Calculate the flattened feature size after conv layers
        self._to_linear = self._get_conv_output(input_size)
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def _get_conv_output(self, shape):
        # Helper to determine the size of the output from conv layers
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.features(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_model(self, train_loader, criterion, optimizer=None, epochs=10, device='cpu'):
        self.to(device)
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    def predict(self, x, device='cpu'):
        self.eval()
        x = x.to(device)
        with torch.no_grad():
            outputs = self(x)
            _, preds = torch.max(outputs, 1)
        return preds, outputs