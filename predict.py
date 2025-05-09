# predict.py

import cv2  # Importa la librería OpenCV para procesamiento de imágenes y video.
import torch  # Importa PyTorch, necesario para trabajar con redes neuronales.
from model import CNNLSTM, ExerciseDataset, transform  # Importa las clases y funciones personalizadas desde el archivo "model.py".

# Configuración de parámetros
SEQUENCE_LENGTH = 30  # Define la longitud de la secuencia de frames (en este caso, 30 frames).
FRAME_SIZE = 224  # Define el tamaño de los frames de entrada (224x224 píxeles).
NUM_CLASSES = 2  # Número de clases para la predicción (por ejemplo, 'Correcto' o 'Incorrecto').
MODEL_PATH = "modelo_entrenado.pth"  # Ruta al archivo del modelo entrenado.

# Ruta del video para predecir
video_path = "videos/nuevo_video.mp4"  # Especifica la ruta al archivo de video del que se hará la predicción.

# Preparar los frames del video
dataset = ExerciseDataset([video_path], [0], transform)  # Crea el conjunto de datos con la ruta del video y las etiquetas correspondientes (en este caso, solo un video).
frames, _ = dataset[0]  # Extrae la secuencia de frames del primer video en el conjunto de datos.
frames = frames.unsqueeze(0)  # Añade una dimensión extra para crear un "batch" de tamaño 1 (el modelo espera un lote de datos).

# Cargar el modelo entrenado
model = CNNLSTM(hidden_size=128, num_classes=NUM_CLASSES)  # Crea una instancia del modelo CNN-LSTM con el tamaño del "hidden layer" y el número de clases.
model.load_state_dict(torch.load(MODEL_PATH))  # Carga los pesos del modelo previamente entrenado desde el archivo especificado.
model.eval()  # Configura el modelo en modo de evaluación (desactiva características específicas de entrenamiento como el dropout).

# Realizar la predicción
with torch.no_grad():  # Desactiva el cálculo de gradientes para ahorrar memoria y tiempo de procesamiento, ya que no necesitamos gradientes para la predicción.
    output = model(frames)  # Pasa los frames al modelo para obtener las predicciones (probabilidades para cada clase).
    prediction = torch.argmax(output, dim=1).item()  # Obtiene la clase con la mayor probabilidad (la predicción final).

# Mostrar el resultado de la predicción
print(f"Predicción: {'Correcto' if prediction == 1 else 'Incorrecto'}")  # Imprime si la predicción es 'Correcto' o 'Incorrecto', según el valor de 'prediction'.
