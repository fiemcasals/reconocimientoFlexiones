# src/extract_pose.py
import cv2
import mediapipe as mp
import os
import json

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

POSE_LANDMARKS = 33  # Total de puntos


def extract_pose_from_video(video_path, output_path, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    poses = []

    while len(poses) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
            poses.append(keypoints)

    cap.release()

    # Padding si hay menos frames
    while len(poses) < max_frames:
        poses.append([0.0] * (POSE_LANDMARKS * 4))

    with open(output_path, 'w') as f:
        json.dump(poses, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Ruta al video .mp4")
    parser.add_argument("output", help="Ruta de salida del JSON")
    args = parser.parse_args()

    extract_pose_from_video(args.video, args.output)
