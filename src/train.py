
import os
import numpy as np
import tensorflow as tf
from preprocessing import preprocess_video
from model import create_model

def load_dataset(dataset_path):
    classes = sorted(os.listdir(dataset_path))
    data, labels = [], []

    for label, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_path, class_name)

        for video in os.listdir(class_folder):
            video_path = os.path.join(class_folder, video)
            frames = preprocess_video(video_path)
            data.append(frames)
            labels.append(label)

    data = np.array(data)
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(classes))

    return data, labels, classes


def train_model(dataset_path):
    data, labels, classes = load_dataset(dataset_path)

    model = create_model((20, 64, 64, 3), len(classes))

    history = model.fit(
        data,
        labels,
        epochs=100,
        batch_size=16,
        validation_split=0.2
    )

    model.save("/kaggle/working/src/final_model.keras")

    return model, history, classes
