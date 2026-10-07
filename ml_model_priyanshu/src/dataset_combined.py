import os
import cv2
import torch
from torch.utils.data import Dataset, random_split
from collections import defaultdict
import random

class CombinedBadmintonDataset(Dataset):
    def __init__(self, samples):
        # samples is a list of tuples: (frame_paths, label, clip_id, source)
        self.samples = samples
        self.classes = ['clear', 'drive', 'drop', 'net', 'smash']
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        frame_paths, label, _, _ = self.samples[index]
        frames = []
        for fp in frame_paths:
            img = cv2.imread(fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            frames.append(img)
        frames = torch.stack(frames)
        return frames, torch.tensor(label, dtype=torch.long)

def load_v1_samples(data_dir=r'C:\batminton\data\processed'):
    classes = ['clear', 'drive', 'drop', 'net', 'smash']
    samples = []
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for video_folder in os.listdir(class_path):
            folder_path = os.path.join(class_path, video_folder)
            if os.path.isdir(folder_path):
                frames = [os.path.join(folder_path, fn) for fn in sorted(os.listdir(folder_path))]
                if len(frames) == 16:
                    samples.append((frames, label, video_folder, 'v1'))
    return samples

def load_part2_samples(data_dir=r'C:\batminton\data\processed_v2'):
    classes = ['clear', 'drive', 'drop', 'net', 'smash']
    samples = []
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for video_folder in sorted(os.listdir(class_path)):
            folder_path = os.path.join(class_path, video_folder)
            if os.path.isdir(folder_path):
                frames = [os.path.join(folder_path, fn) for fn in sorted(os.listdir(folder_path))]
                if len(frames) == 16:
                    samples.append((frames, label, video_folder, 'part2'))
    return samples

def build_combined_splits(seed=42):
    v1_samples = load_v1_samples()
    part2_samples = load_part2_samples()
    
    # 1. Exact V1 holdout test set (204 clips)
    v1_total = len(v1_samples)
    v1_train_size = int(0.8 * v1_total)
    v1_test_size = v1_total - v1_train_size
    g_v1 = torch.Generator().manual_seed(seed)
    v1_ds = CombinedBadmintonDataset(v1_samples)
    v1_tr_split, v1_te_split = random_split(v1_ds, [v1_train_size, v1_test_size], generator=g_v1)
    
    v1_train_samples = [v1_samples[i] for i in v1_tr_split.indices]
    v1_test_samples = [v1_samples[i] for i in v1_te_split.indices]
    
    # 2. Stratified split of PART 2 (2,440 clips)
    part2_class_indices = defaultdict(list)
    for idx, (_, label, _, _) in enumerate(part2_samples):
        part2_class_indices[label].append(idx)
        
    p2_train_indices = []
    p2_val_indices = []
    p2_test_indices = []
    
    rng = random.Random(seed)
    for label in sorted(part2_class_indices.keys()):
        idxs = list(part2_class_indices[label])
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = int(0.15 * n)
        n_test = int(0.15 * n)
        n_train = n - n_val - n_test
        
        p2_train_indices.extend(idxs[:n_train])
        p2_val_indices.extend(idxs[n_train:n_train + n_val])
        p2_test_indices.extend(idxs[n_train + n_val:])
        
    p2_train_samples = [part2_samples[i] for i in p2_train_indices]
    p2_val_samples = [part2_samples[i] for i in p2_val_indices]
    p2_test_samples = [part2_samples[i] for i in p2_test_indices]
    
    # Combined Training Set
    combined_train_samples = v1_train_samples + p2_train_samples
    
    return {
        'train': CombinedBadmintonDataset(combined_train_samples),
        'val': CombinedBadmintonDataset(p2_val_samples),
        'v1_test': CombinedBadmintonDataset(v1_test_samples),
        'part2_test': CombinedBadmintonDataset(p2_test_samples)
    }

if __name__ == '__main__':
    splits = build_combined_splits()
    print('Splits built successfully!')
    tr_len = len(splits['train'])
    val_len = len(splits['val'])
    v1_len = len(splits['v1_test'])
    p2_len = len(splits['part2_test'])
    print(f'Train clips       : {tr_len}')
    print(f'Val clips         : {val_len}')
    print(f'V1 Test clips     : {v1_len}')
    print(f'PART 2 Test clips : {p2_len}')
    print(f'Total clips       : {tr_len + val_len + v1_len + p2_len}')
