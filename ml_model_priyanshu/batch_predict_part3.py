import os
import sys
import csv
import time
import cv2
import torch
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = r"C:\batminton"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import CNNLSTM

MODEL_PATH = os.path.join(PROJECT_ROOT, "badminton_cnn_lstm.pth")
DATA_DIR = r"C:\Users\priya\Downloads\PART 3"
PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "PART3_predictions.csv")
SUMMARY_TXT = os.path.join(PROJECT_ROOT, "PART3_summary.txt")
MISCLASSIFIED_CSV = os.path.join(PROJECT_ROOT, "PART3_misclassified.csv")

MODEL_CLASSES = ["Clear", "Drive", "Drop", "Net Shot", "Smash"]
FOLDER_TO_CLASS = {
    "CLEAR": "Clear",
    "DROP": "Drop",
    "NET SHOT": "Net Shot",
    "SMASH": "Smash"
}

# ImageNet normalization tensors
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

def extract_16_frames(video_path, device):
    """
    Extracts 16 consecutive frames matching training preprocessing:
    BGR -> RGB, 224x224, [0, 1], ImageNet normalized.
    Returns: tensor [1, 16, 3, 224, 224] on device, or (None, error_message).
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Unable to open video file"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total_frames == 16:
            for _ in range(16):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames.append(frame)
        elif total_frames > 16:
            start_idx = (total_frames - 16) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            for _ in range(16):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames.append(frame)
        cap.release()

        # Fallback if position set did not yield 16 valid frames
        if len(frames) < 16:
            cap = cv2.VideoCapture(video_path)
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                all_frames.append(frame)
            cap.release()

            if len(all_frames) >= 16:
                start_idx = (len(all_frames) - 16) // 2
                frames = all_frames[start_idx : start_idx + 16]
            else:
                return None, f"Video has only {len(all_frames)} readable frames (< 16 required)"

        if len(frames) != 16:
            return None, f"Extracted {len(frames)} frames (< 16 required)"

        # Preprocessing matching training
        tensor_list = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224))
            norm = resized.astype(np.float32) / 255.0
            t = torch.from_numpy(norm).permute(2, 0, 1)  # [3, 224, 224]
            t = (t - MEAN) / STD
            tensor_list.append(t)

        seq = torch.stack(tensor_list).unsqueeze(0).to(device)  # [1, 16, 3, 224, 224]
        return seq, None

    except Exception as e:
        return None, str(e)


def main():
    print("=" * 60)
    print("BADMINTON CNN-LSTM BATCH PREDICTION — PART 3")
    print("=" * 60)

    # 1. Verify Model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    # 2. Verify CUDA & Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 3. Load Model in Eval Mode
    print("Loading existing CNN-LSTM model...")
    model = CNNLSTM(num_classes=5)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully in eval mode. (Output classes = 5)")

    # 4. Discover Videos
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Input directory not found at {DATA_DIR}")
        sys.exit(1)

    detected_folders = [f for f in ["CLEAR", "DROP", "NET SHOT", "SMASH"] if os.path.isdir(os.path.join(DATA_DIR, f))]
    folder_video_counts = {}
    video_tasks = []

    for folder in detected_folders:
        folder_path = os.path.join(DATA_DIR, folder)
        vids = []
        for item in sorted(os.listdir(folder_path)):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                sub_vids = [
                    (folder, os.path.join(item_path, f), f)
                    for f in sorted(os.listdir(item_path))
                    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
                ]
                vids.extend(sub_vids)
            elif os.path.isfile(item_path) and item.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                vids.append((folder, item_path, item))
        folder_video_counts[folder] = len(vids)
        video_tasks.extend(vids)

    total_videos = len(video_tasks)
    print(f"\nFolders detected: {len(detected_folders)}")
    for f, count in folder_video_counts.items():
        print(f"  - {f}: {count} videos")
    print(f"Total videos to process: {total_videos}\n")

    # CSV headers
    pred_fieldnames = [
        "Video",
        "Actual_Folder",
        "Predicted_Class",
        "Confidence",
        "Clear_Probability",
        "Drive_Probability",
        "Drop_Probability",
        "NetShot_Probability",
        "Smash_Probability",
        "Status"
    ]

    misc_fieldnames = [
        "Video",
        "Actual_Folder",
        "Actual_Class",
        "Predicted_Class",
        "Confidence",
        "Clear_Probability",
        "Drive_Probability",
        "Drop_Probability",
        "NetShot_Probability",
        "Smash_Probability",
        "Status"
    ]

    # Initialize CSV files
    f_pred = open(PREDICTIONS_CSV, mode="w", newline="", encoding="utf-8")
    writer_pred = csv.DictWriter(f_pred, fieldnames=pred_fieldnames)
    writer_pred.writeheader()
    f_pred.flush()

    f_misc = open(MISCLASSIFIED_CSV, mode="w", newline="", encoding="utf-8")
    writer_misc = csv.DictWriter(f_misc, fieldnames=misc_fieldnames)
    writer_misc.writeheader()
    f_misc.flush()

    # Tracking metrics
    successful_count = 0
    failed_count = 0
    confidence_sum = 0.0
    failed_videos = []

    model_class_counts = {c: 0 for c in MODEL_CLASSES}
    folder_prediction_matrix = {f: {c: 0 for c in MODEL_CLASSES} for f in detected_folders}

    start_time = time.time()

    # Process videos sequentially
    for idx, (folder, full_path, vfilename) in enumerate(video_tasks, 1):
        tensor, err = extract_16_frames(full_path, device)

        if err is not None:
            failed_count += 1
            failed_videos.append((vfilename, folder, err))
            row = {
                "Video": vfilename,
                "Actual_Folder": folder,
                "Predicted_Class": "N/A",
                "Confidence": "N/A",
                "Clear_Probability": "N/A",
                "Drive_Probability": "N/A",
                "Drop_Probability": "N/A",
                "NetShot_Probability": "N/A",
                "Smash_Probability": "N/A",
                "Status": "FAILED"
            }
            writer_pred.writerow(row)
            f_pred.flush()

            print(f"Processing {idx}/{total_videos}")
            print(f"Folder: {folder}")
            print(f"Video: {vfilename}")
            print(f"Status: FAILED ({err})\n", flush=True)
            continue

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            conf_val = probs[pred_idx].item() * 100.0
            pred_class = MODEL_CLASSES[pred_idx]
            prob_floats = [p.item() for p in probs]

        successful_count += 1
        confidence_sum += conf_val
        model_class_counts[pred_class] += 1
        folder_prediction_matrix[folder][pred_class] += 1

        conf_str = f"{conf_val:.2f}%"
        clear_prob = f"{prob_floats[0]:.4f}"
        drive_prob = f"{prob_floats[1]:.4f}"
        drop_prob = f"{prob_floats[2]:.4f}"
        net_prob = f"{prob_floats[3]:.4f}"
        smash_prob = f"{prob_floats[4]:.4f}"

        row = {
            "Video": vfilename,
            "Actual_Folder": folder,
            "Predicted_Class": pred_class,
            "Confidence": conf_str,
            "Clear_Probability": clear_prob,
            "Drive_Probability": drive_prob,
            "Drop_Probability": drop_prob,
            "NetShot_Probability": net_prob,
            "Smash_Probability": smash_prob,
            "Status": "SUCCESS"
        }
        writer_pred.writerow(row)

        actual_class = FOLDER_TO_CLASS.get(folder, folder)
        if actual_class != pred_class:
            misc_row = {
                "Video": vfilename,
                "Actual_Folder": folder,
                "Actual_Class": actual_class,
                "Predicted_Class": pred_class,
                "Confidence": conf_str,
                "Clear_Probability": clear_prob,
                "Drive_Probability": drive_prob,
                "Drop_Probability": drop_prob,
                "NetShot_Probability": net_prob,
                "Smash_Probability": smash_prob,
                "Status": "SUCCESS"
            }
            writer_misc.writerow(misc_row)

        if idx % 10 == 0:
            f_pred.flush()
            f_misc.flush()

        print(f"Processing {idx}/{total_videos}")
        print(f"Folder: {folder}")
        print(f"Video: {vfilename}")
        print(f"Prediction: {pred_class}")
        print(f"Confidence: {conf_str}\n", flush=True)

    f_pred.close()
    f_misc.close()

    elapsed = time.time() - start_time
    avg_conf = (confidence_sum / successful_count) if successful_count > 0 else 0.0

    # Write Summary File
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f_sum:
        f_sum.write("=" * 60 + "\n")
        f_sum.write("       BADMINTON CNN-LSTM PART 3 PREDICTION SUMMARY\n")
        f_sum.write("=" * 60 + "\n\n")

        f_sum.write(f"Number of Folders Detected: {len(detected_folders)}\n")
        f_sum.write(f"Folder Names: {', '.join(detected_folders)}\n\n")

        f_sum.write("Videos Found per Folder:\n")
        for f, cnt in folder_video_counts.items():
            f_sum.write(f"  - {f}: {cnt} videos\n")
        f_sum.write("\n")

        f_sum.write(f"Total Videos Processed: {total_videos}\n")
        f_sum.write(f"Successful Predictions: {successful_count}\n")
        f_sum.write(f"Failed Predictions: {failed_count}\n")
        f_sum.write(f"Average Confidence: {avg_conf:.2f}%\n")
        f_sum.write(f"Total Processing Time: {elapsed:.2f}s ({total_videos / elapsed:.1f} vps)\n\n")

        f_sum.write("-" * 50 + "\n")
        f_sum.write("Overall Prediction Count by Model Class (5 Classes):\n")
        f_sum.write("-" * 50 + "\n")
        for c in MODEL_CLASSES:
            f_sum.write(f"  - {c:<10}: {model_class_counts[c]} ({model_class_counts[c] / successful_count * 100:.2f}%)\n" if successful_count > 0 else f"  - {c:<10}: {model_class_counts[c]}\n")
        f_sum.write("\n")

        f_sum.write("-" * 50 + "\n")
        f_sum.write("Per-Folder Prediction Breakdown:\n")
        f_sum.write("-" * 50 + "\n")
        for f in detected_folders:
            act_cls = FOLDER_TO_CLASS.get(f, f)
            f_sum.write(f"Folder: {f} (Actual Mapped Class: {act_cls}) [Total: {folder_video_counts[f]}]\n")
            for c in MODEL_CLASSES:
                cnt = folder_prediction_matrix[f][c]
                pct = (cnt / folder_video_counts[f] * 100.0) if folder_video_counts[f] > 0 else 0.0
                f_sum.write(f"    -> Predicted as {c:<10}: {cnt} ({pct:.2f}%)\n")
            f_sum.write("\n")

        f_sum.write("-" * 50 + "\n")
        f_sum.write(f"Failed Videos List ({failed_count}):\n")
        f_sum.write("-" * 50 + "\n")
        if failed_count == 0:
            f_sum.write("  None. All videos processed successfully.\n")
        else:
            for vfile, fld, reason in failed_videos:
                f_sum.write(f"  - Video: {vfile} | Folder: {fld} | Reason: {reason}\n")
        f_sum.write("\n" + "=" * 60 + "\n")

    # Print Final Terminal Output
    print("=" * 60)
    print("                 BATCH PREDICTION COMPLETED")
    print("=" * 60)
    print(f"Total videos           : {total_videos}")
    print(f"Successful predictions : {successful_count}")
    print(f"Failed predictions     : {failed_count}")
    print(f"Average confidence     : {avg_conf:.2f}%")
    print(f"Elapsed Time           : {elapsed:.2f} seconds")
    print("-" * 60)
    print("Prediction Distribution:")
    for c in MODEL_CLASSES:
        cnt = model_class_counts[c]
        pct = (cnt / successful_count * 100.0) if successful_count > 0 else 0.0
        print(f"  {c:<10}: {cnt:>5} ({pct:5.2f}%)")
    print("-" * 60)
    print(f"Output CSV             : {PREDICTIONS_CSV}")
    print(f"Summary TXT            : {SUMMARY_TXT}")
    print(f"Misclassified CSV      : {MISCLASSIFIED_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    main()
