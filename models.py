# model.py

# Librerías necesarias
import cv2  # para leer videos
import numpy as np  # para trabajar con arrays
import torch  # para usar tensores y redes neuronales
import torch.nn as nn  # para construir capas de red
from torchvision import models, transforms  # modelos y transformaciones prehechas
from torch.utils.data import Dataset  # clase base para crear datasets personalizados

# Transformaciones para aplicar a cada imagen del video
transform = transforms.Compose([
    transforms.ToPILImage(),  # convierte la imagen a formato PIL
    transforms.Resize((224, 224)),  # redimensiona la imagen a 224x224 (lo que espera resnet18)
    transforms.ToTensor(),  # convierte la imagen a tensor (PyTorch)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normaliza con la media del dataset ImageNet
                         std=[0.229, 0.224, 0.225])  # normaliza con la desviación estándar del dataset ImageNet
])

# Clase para crear un dataset personalizado a partir de videos
class ExerciseDataset(Dataset):
    def __init__(self, video_paths, labels, transform):
        self.video_paths = video_paths  # rutas de los videos
        self.labels = labels  # etiquetas correspondientes a cada video
        self.transform = transform  # transformaciones a aplicar a los frames

    def __len__(self):
        return len(self.video_paths)  # cantidad total de videos

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]  # obtiene la ruta del video en el índice idx
        label = self.labels[idx]  # obtiene la etiqueta correspondiente
        frames = self.extract_frames(video_path)  # extrae 30 frames del video
        return frames, label  # retorna los frames (como tensor) y su etiqueta

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)  # abre el video
        frames = []  # lista para guardar los frames procesados
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # cantidad total de frames en el video
        frame_indices = np.linspace(0, total_frames - 1, 30, dtype=int)  # elige 30 frames espaciados

        for i in frame_indices:  # recorre los índices seleccionados
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # salta al frame i
            ret, frame = cap.read()  # lee el frame
            if not ret:
                break  # si no se puede leer, se sale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convierte de BGR a RGB
            frame = self.transform(frame)  # aplica las transformaciones
            frames.append(frame)  # agrega el frame procesado a la lista

        cap.release()  # cierra el video
        frames = torch.stack(frames)  # convierte la lista de frames en un solo tensor
        return frames  # devuelve el tensor de frames

# Clase que define el modelo CNN + LSTM
class CNNLSTM(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(CNNLSTM, self).__init__()  # inicializa la clase base nn.Module
        self.cnn = models.resnet18(pretrained=True)  # carga ResNet18 preentrenada (sin modificar)
        self.cnn.fc = nn.Identity()  # quita la última capa (fc) de ResNet para usar solo las features
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)  
        # LSTM que toma la salida de ResNet (512 valores) y devuelve secuencias procesadas
        self.fc = nn.Linear(hidden_size, num_classes)  # capa final que clasifica en num_classes clases

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()  # obtiene las dimensiones del input (batch, secuencia, canales, alto, ancho)
        c_in = x.view(batch_size * seq_len, C, H, W)  # aplana para pasar cada frame por separado por la CNN
        c_out = self.cnn(c_in)  # extrae características de cada frame con la CNN
        r_in = c_out.view(batch_size, seq_len, -1)  # vuelve a agrupar las salidas por secuencia para el LSTM
        r_out, _ = self.lstm(r_in)  # pasa la secuencia por el LSTM
        out = self.fc(r_out[:, -1, :])  # toma la última salida del LSTM y la pasa por la capa final
        return out  # devuelve la predicción
