# src/predict_pose.py
import torch
import json
from model_lstm import PoseLSTM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Cargar modelo
model = PoseLSTM()
model.load_state_dict(torch.load("models/modelo_lstm_pose.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Cargar pose desde archivo
def predict_pose(json_path):
    with open(json_path, 'r') as f:
        sequence = json.load(f)
    tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    output = model(tensor)
    pred = torch.argmax(output, dim=1).item()
    return "correcto" if pred == 1 else "incorrecto"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Ruta al archivo .json de pose")
    args = parser.parse_args()

    resultado = predict_pose(args.json_path)
    print("Resultado:", resultado)
