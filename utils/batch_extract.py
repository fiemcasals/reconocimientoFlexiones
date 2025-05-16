# utils/batch_extract.py
import os
from src.extract_pose import extract_pose_from_video

INPUT_DIR = "dataset"
OUTPUT_DIR = "poses"

for label in ["correcto", "incorrecto"]:
    in_path = os.path.join(INPUT_DIR, label)
    out_path = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out_path, exist_ok=True)

    for file in os.listdir(in_path):
        if file.endswith(".mp4"):
            video_path = os.path.join(in_path, file)
            json_name = os.path.splitext(file)[0] + ".json"
            json_path = os.path.join(out_path, json_name)

            print(f"Procesando {video_path}...")
            extract_pose_from_video(video_path, json_path)

print("✅ Extracción masiva completa.")
