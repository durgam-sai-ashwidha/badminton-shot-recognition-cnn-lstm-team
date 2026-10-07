import os
import cv2
import torch
from torch.utils.data import Dataset


class BadmintonDataset(Dataset):

    def __init__(self, data_dir):
        self.samples = []
        self.classes = ["clear", "drive", "drop", "net", "smash"]

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)

            for video_folder in os.listdir(class_path):
                folder_path = os.path.join(class_path, video_folder)

                if os.path.isdir(folder_path):
                    frames = []

                    for frame_name in sorted(os.listdir(folder_path)):
                        if frame_name.lower().endswith((".jpg", ".jpeg", ".png")):
                            frame_path = os.path.join(folder_path, frame_name)
                            frames.append(frame_path)

                    if len(frames) == 16:
                        self.samples.append((frames, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_paths, label = self.samples[index]

        frames = []

        for frame_path in frame_paths:
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = image / 255.0

            image = torch.tensor(image, dtype=torch.float32)
            image = image.permute(2, 0, 1)

            mean = torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32
            ).view(3, 1, 1)

            std = torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32
            ).view(3, 1, 1)

            image = (image - mean) / std

            frames.append(image)

        frames = torch.stack(frames)

        return frames, torch.tensor(label, dtype=torch.long)