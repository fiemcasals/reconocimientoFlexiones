# train.py

import os  # Importa la librería 'os' para interactuar con el sistema de archivos.
import cv2  # Importa OpenCV para procesamiento de imágenes y video.
import torch  # Importa PyTorch para trabajar con redes neuronales.
import numpy as np  # Importa Numpy, útil para manipulaciones de arrays y operaciones matemáticas.
import torch.nn as nn  # Importa módulos de redes neuronales de PyTorch.
from torchvision import models, transforms  # Importa modelos pre-entrenados y transformaciones de imágenes.
from torch.utils.data import Dataset, DataLoader  # Importa clases necesarias para crear y cargar conjuntos de datos.
from model import CNNLSTM, ExerciseDataset, transform  # Importa las clases y funciones personalizadas desde el archivo "model.py".

# Configuración de parámetros
SEQUENCE_LENGTH = 30  # Longitud de la secuencia de frames (30 frames por secuencia).
FRAME_SIZE = 224  # Tamaño de los frames de entrada (224x224 píxeles).
BATCH_SIZE = 8  # Número de ejemplos por batch durante el entrenamiento.
NUM_CLASSES = 2  # Número de clases en la predicción (por ejemplo, 'Correcto' o 'Incorrecto').
EPOCHS = 10  # Número de épocas (iteraciones sobre todo el conjunto de datos) para entrenar el modelo.
MODEL_PATH = "modelo_entrenado.pth"  # Ruta donde se guardará el modelo entrenado.

# Rutas de los videos y sus etiquetas
video_paths = ['videos/correcto1.mp4', 'videos/incorrecto1.mp4']  # Lista de rutas a los archivos de video para entrenamiento.
labels = [1, 0]  # Etiquetas correspondientes a los videos (1 para 'Correcto', 0 para 'Incorrecto').

# Crear el conjunto de datos y el DataLoader
dataset = ExerciseDataset(video_paths, labels, transform)  # Crea un objeto 'ExerciseDataset' para cargar los videos y sus etiquetas.
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # Crea un DataLoader para cargar el conjunto de datos en batches de tamaño 'BATCH_SIZE' y con aleatorización.

# Inicializar el modelo, la función de pérdida y el optimizador
model = CNNLSTM(hidden_size=128, num_classes=NUM_CLASSES)  # Crea una instancia del modelo CNN-LSTM con tamaño de capa oculta y número de clases especificado.
criterion = nn.CrossEntropyLoss()  # Define la función de pérdida, en este caso, la pérdida de entropía cruzada, que se usa para clasificación.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Crea el optimizador Adam con una tasa de aprendizaje de 0.0001 para actualizar los parámetros del modelo.

# Entrenamiento del modelo
model.train()  # Establece el modelo en modo de entrenamiento (habilita ciertas características como el dropout).
for epoch in range(EPOCHS):  # Itera a través de las épocas de entrenamiento.
    total_loss = 0.0  # Inicializa la variable para acumular la pérdida total de cada época.
    for frames, labels in dataloader:  # Itera sobre cada batch de frames y etiquetas del DataLoader.
        outputs = model(frames)  # Pasa los frames a través del modelo para obtener las predicciones.
        loss = criterion(outputs, labels)  # Calcula la pérdida comparando las predicciones con las etiquetas verdaderas.
        optimizer.zero_grad()  # Restaura los gradientes acumulados de la iteración anterior.
        loss.backward()  # Calcula los gradientes de la pérdida con respecto a los parámetros del modelo.
        optimizer.step()  # Actualiza los parámetros del modelo con los gradientes calculados.
        total_loss += loss.item()  # Acumula la pérdida de la iteración.
    print(f"Época {epoch+1}/{EPOCHS} - Pérdida: {total_loss:.4f}")  # Imprime la pérdida total al final de cada época.

# Guardar el modelo entrenado
torch.save(model.state_dict(), MODEL_PATH)  # Guarda los pesos del modelo entrenado en el archivo especificado.
print(f"Modelo guardado en {MODEL_PATH}")  # Imprime un mensaje indicando que el modelo fue guardado.


"""Este script entrena un modelo CNN-LSTM utilizando un conjunto de videos etiquetados, calcula la pérdida y ajusta los parámetros del modelo durante un número determinado de épocas. Al final, guarda los pesos del modelo entrenado en un archivo."""