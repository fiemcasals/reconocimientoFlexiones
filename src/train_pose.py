# src/train_pose.py
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset_pose import PoseSequenceDataset
from model_lstm import PoseLSTM
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

pose_dir = './poses'
batch_size = 8
epochs = 20

# Dataset y DataLoader
dataset = PoseSequenceDataset(pose_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Modelo y pérdida
model = PoseLSTM().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Entrenamiento
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in dataloader:
        sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Época {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Acc: {acc:.4f}")

# Guardar modelo
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/modelo_lstm_pose.pth")
print("Modelo guardado en models/modelo_lstm_pose.pth")
