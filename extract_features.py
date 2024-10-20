import shutil   #for file operation
import numpy as np
import cv2   #for computer vision operation
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
import config


def extract_frames(video):
    path = os.path.join(config.train_path, 'temporary_images')
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    video_path = os.path.join(config.train_path, 'video', video)
    cap = cv2.VideoCapture(video_path)
    count = 0
    image_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(path, "frame%d.jpg" % count), frame)
        image_list.append(os.path.join(path, "frame%d.jpg" % count))
        count += 1
    cap.release()
    return image_list


def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    return img


def extract_features(video):
    video_id = video.split(".")[0]  # Extracting video ID
    print("Processing video:", video_id)  # Printing video ID
    image_list = extract_frames(video)
    samples = np.round(np.linspace(0, len(image_list) - 1, 80))
    image_list = [image_list[int(sample)] for sample in samples]
    images = np.array([load_image(img) for img in image_list])
    return images


def extract_feats_pretrained_cnn():
    if not os.path.isdir(os.path.join(config.train_path, 'feat')):
        os.mkdir(os.path.join(config.train_path, 'feat'))

    video_list = os.listdir(os.path.join(config.train_path, 'video'))

    # Remove any system files, like '.ipynb_checkpoints'
    video_list = [video for video in video_list if not video.startswith('.')]

    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False)
    # Remove the classification layers
    model = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))

    for video in video_list:
        outfile = os.path.join(config.train_path, 'feat', video + '.npy')
        img_feats = extract_features(video)
        # Extract features using VGG16
        features = model.predict(img_feats)
        np.save(outfile, features)


if __name__ == "__main__":
    extract_feats_pretrained_cnn()
