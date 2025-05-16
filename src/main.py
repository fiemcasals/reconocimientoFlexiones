# main.py
import os
import uuid
from extract_pose import extract_pose_from_video
from predict_pose import predict_pose

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Ruta al video de entrada")
    args = parser.parse_args()

    # Generar nombre temporal para el archivo JSON
    json_name = f"tmp_{uuid.uuid4().hex}.json"

    # Extraer pose
    extract_pose_from_video(args.video_path, json_name)

    # Predecir clase
    resultado = predict_pose(json_name)
    print("Resultado final:", resultado)

    # Limpiar archivo temporal
    os.remove(json_name)
