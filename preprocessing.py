
import cv2
import numpy as np
import random

IMG_SIZE = 64
SEQUENCE_LENGTH = 20

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // SEQUENCE_LENGTH)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0

        if random.random() < 0.2:
            frame = cv2.flip(frame, 1)

        frames.append(frame)

    cap.release()

    while len(frames) < SEQUENCE_LENGTH:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    return np.array(frames[:SEQUENCE_LENGTH])
