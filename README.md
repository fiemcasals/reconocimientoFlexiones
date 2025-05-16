# Clasificador de Ejercicios Físicos con Pose Estimation y LSTM

Este proyecto utiliza MediaPipe para extraer poses humanas desde videos y una red neuronal LSTM para clasificar la calidad de la ejecución (correcta o incorrecta) de un ejercicio físico.

---

## 📁 Estructura del proyecto

```bash
.
├── dataset/           # Videos divididos en "correcto" e "incorrecto"
├── poses/             # Archivos .json generados desde videos (pose por frame)
├── models/            # Modelos entrenados (.pth)
├── src/               # Código fuente
│   ├── extract_pose.py
│   ├── dataset_pose.py
│   ├── model_lstm.py
│   ├── train_pose.py
│   ├── predict_pose.py
│   ├── main.py
├── utils/             # Utilidades auxiliares
│   └── batch_extract.py
├── requirements.txt   # Dependencias
└── README.md          # Este archivo
```

---

## ⚙️ Requisitos

- Python 3.8+
- pip

---

## 🧪 Instalación (entorno virtual recomendado)

```bash
# Clonar el repositorio
https://github.com/tu_usuario/ejercicio_ia.git
cd ejercicio_ia

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🏋️ Entrenamiento del modelo

1. Colocá tus videos en `dataset/correcto/` y `dataset/incorrecto/`.
2. Extraé las poses a `.json`:

```bash
python src/extract_pose.py dataset/correcto/ej1.mp4 poses/correcto/ej1.json
```

O extraé todos los videos de una sola vez:

```bash
python utils/batch_extract.py
```

3. Entrená el modelo:

```bash
python src/train_pose.py
```

---

## 🔍 Clasificación de un nuevo video

```bash
python src/main.py mi_video.mp4
```

Este comando:
- Extrae automáticamente las poses con MediaPipe
- Clasifica con el modelo entrenado
- Devuelve "correcto" o "incorrecto"

---

## 🧠 Tecnologías usadas

- Python + PyTorch
- MediaPipe (detección de pose)
- OpenCV (procesamiento de video)
- LSTM (análisis temporal)

---
