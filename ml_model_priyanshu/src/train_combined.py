import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from dataset_combined import build_combined_splits
from model import CNNLSTM

ROOT_DIR = os.path.dirname(SRC_DIR)
V1_MODEL_PATH = os.path.join(ROOT_DIR, 'badminton_cnn_lstm.pth')
TEMP_IMPROVED_MODEL_PATH = os.path.join(ROOT_DIR, 'badminton_cnn_lstm_improved_temp.pth')

BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 0.00005
SEED = 42
CLASS_NAMES = ['Clear', 'Drive', 'Drop', 'Net Shot', 'Smash']

def evaluate_subset(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    tot_loss = 0.0
    tot_samples = 0
    crit = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = crit(outputs, labels)
            
            tot_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            tot_samples += len(labels)
            
    acc = accuracy_score(all_labels, all_preds) * 100.0
    avg_loss = tot_loss / tot_samples if tot_samples > 0 else 0.0
    return avg_loss, acc, all_labels, all_preds

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('=' * 65, flush=True)
    print('   CONTINUED TRAINING: FINE-TUNING EXISTING CNN-LSTM MODEL', flush=True)
    print('=' * 65, flush=True)
    print(f'PyTorch Version      : {torch.__version__}', flush=True)
    print(f'Using Device         : {device}', flush=True)
    if torch.cuda.is_available():
        print(f'GPU Device Name      : {torch.cuda.get_device_name(0)}', flush=True)
        print(f'GPU Memory           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB', flush=True)
    
    print(f'Base Model to Fine-Tune : {V1_MODEL_PATH}', flush=True)
    print(f'Temporary Target Output : {TEMP_IMPROVED_MODEL_PATH}', flush=True)
    print(f'Hyperparameters         : Batch Size={BATCH_SIZE}, LR={LEARNING_RATE}, Epochs={EPOCHS}', flush=True)
    print('-' * 65, flush=True)
    
    # 1. Load splits
    print('Preparing combined dataset splits (seed=42)...', flush=True)
    splits = build_combined_splits(seed=SEED)
    
    train_ds = splits['train']
    val_ds = splits['val']
    v1_test_ds = splits['v1_test']
    p2_test_ds = splits['part2_test']
    
    print(f'Training Clips       : {len(train_ds)} (816 V1 + 1,708 PART 2)', flush=True)
    print(f'Validation Clips     : {len(val_ds)} (held-out from PART 2)', flush=True)
    print(f'V1 Test Benchmark    : {len(v1_test_ds)} clips (Original V1 holdout)', flush=True)
    print(f'PART 2 Test Benchmark: {len(p2_test_ds)} clips (Held-out PART 2 set)', flush=True)
    print(f'Total Unique Clips   : {len(train_ds) + len(val_ds) + len(v1_test_ds) + len(p2_test_ds)}', flush=True)
    print('-' * 65, flush=True)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
    v1_test_loader = DataLoader(v1_test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
    p2_test_loader = DataLoader(p2_test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
    
    # 2. Load existing V1 model weights
    print(f'Loading existing V1 weights from: {V1_MODEL_PATH}', flush=True)
    model = CNNLSTM(num_classes=5)
    model.load_state_dict(torch.load(V1_MODEL_PATH, map_location=device))
    model = model.to(device)
    print('V1 model loaded successfully! Preserving existing weights.', flush=True)
    
    # Initial validation before any fine-tuning
    init_val_loss, init_val_acc, _, _ = evaluate_subset(model, val_loader, device)
    init_v1_loss, init_v1_acc, _, _ = evaluate_subset(model, v1_test_loader, device)
    print(f'Initial Check: Val Acc on PART 2 Val={init_val_acc:.2f}%, Test Acc on V1 Test={init_v1_acc:.2f}%', flush=True)
    print('-' * 65, flush=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = init_val_acc
    best_val_loss = init_val_loss
    best_epoch = 0
    training_start = time.time()
    
    print('Starting fine-tuning across all epochs...', flush=True)
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (frames, labels) in enumerate(train_loader, start=1):
            frames, labels = frames.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
            
        epoch_tr_loss = train_loss / train_total
        epoch_tr_acc = (train_correct / train_total) * 100.0
        
        val_loss, val_acc, _, _ = evaluate_subset(model, val_loader, device)
        epoch_duration = time.time() - epoch_start
        
        is_best = False
        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            is_best = True
            torch.save(model.state_dict(), TEMP_IMPROVED_MODEL_PATH)
            
        star = ' [*SAVED BEST TEMP*]' if is_best else ''
        print(f'Epoch [{epoch:02d}/{EPOCHS:02d}] ({epoch_duration:4.1f}s) | '
              f'Train Loss: {epoch_tr_loss:.4f}, Train Acc: {epoch_tr_acc:6.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:6.2f}%{star}', flush=True)
              
    total_time = time.time() - training_start
    print('=' * 65, flush=True)
    print('           FINE-TUNING PHASE COMPLETED', flush=True)
    print('=' * 65, flush=True)
    print(f'Total Fine-Tuning Time: {total_time / 60:.2f} minutes ({total_time:.1f}s)', flush=True)
    print(f'Best Epoch            : Epoch {best_epoch}', flush=True)
    print(f'Best Val Accuracy     : {best_val_acc:.2f}%', flush=True)
    print(f'Best Val Loss         : {best_val_loss:.4f}', flush=True)
    print(f'Temp Improved Checkpoint: {TEMP_IMPROVED_MODEL_PATH}', flush=True)
    print('-' * 65, flush=True)
    
    # 3. Comprehensive Evaluation of Best Temp Checkpoint
    if os.path.exists(TEMP_IMPROVED_MODEL_PATH):
        print('Loading best temporary checkpoint for evaluation...', flush=True)
        model.load_state_dict(torch.load(TEMP_IMPROVED_MODEL_PATH, map_location=device))
    model.eval()
    
    print('\n' + '=' * 65, flush=True)
    print('     BENCHMARK 1: EVALUATION ON ORIGINAL V1 TEST SET (204 clips)', flush=True)
    print('=' * 65, flush=True)
    _, v1_acc, v1_true, v1_pred = evaluate_subset(model, v1_test_loader, device)
    print(f'V1 Test Accuracy (Improved Model): {v1_acc:.2f}% (Original V1 was 96.57%)', flush=True)
    print(classification_report(v1_true, v1_pred, target_names=CLASS_NAMES, digits=4, zero_division=0), flush=True)
    
    print('\n' + '=' * 65, flush=True)
    print('     BENCHMARK 2: EVALUATION ON HELD-OUT PART 2 TEST SET (366 clips)', flush=True)
    print('=' * 65, flush=True)
    _, p2_acc, p2_true, p2_pred = evaluate_subset(model, p2_test_loader, device)
    print(f'PART 2 Held-Out Test Accuracy    : {p2_acc:.2f}%', flush=True)
    print(classification_report(p2_true, p2_pred, target_names=CLASS_NAMES, digits=4, zero_division=0), flush=True)
    
    print('Confusion Matrix (PART 2 Test Set):', flush=True)
    cm = confusion_matrix(p2_true, p2_pred)
    col_hdr = '%-12s ' % 'True \\ Pred' + ' '.join(['%8s' % n[:8] for n in CLASS_NAMES])
    print(col_hdr, flush=True)
    print('-' * len(col_hdr), flush=True)
    for i, row in enumerate(cm):
        print('%-12s ' % CLASS_NAMES[i] + ' '.join(['%8d' % v for v in row]), flush=True)
        
    print('\n' + '=' * 65, flush=True)
    print('             COMBINED OVERALL TEST PERFORMANCE (570 clips)', flush=True)
    print('=' * 65, flush=True)
    combined_true = v1_true + p2_true
    combined_pred = v1_pred + p2_pred
    comb_acc = accuracy_score(combined_true, combined_pred) * 100.0
    comb_p_macro = precision_score(combined_true, combined_pred, average='macro', zero_division=0) * 100.0
    comb_r_macro = recall_score(combined_true, combined_pred, average='macro', zero_division=0) * 100.0
    comb_f1_macro = f1_score(combined_true, combined_pred, average='macro', zero_division=0) * 100.0
    comb_f1_weighted = f1_score(combined_true, combined_pred, average='weighted', zero_division=0) * 100.0
    
    print(f'Total Test Samples       : {len(combined_true)} (204 V1 + 366 PART 2)', flush=True)
    print(f'Overall Test Accuracy    : {comb_acc:.2f}%', flush=True)
    print(f'Overall Macro F1 Score   : {comb_f1_macro:.2f}%', flush=True)
    print(f'Overall Weighted F1 Score: {comb_f1_weighted:.2f}%', flush=True)
    print(classification_report(combined_true, combined_pred, target_names=CLASS_NAMES, digits=4, zero_division=0), flush=True)
    print('=' * 65, flush=True)

if __name__ == '__main__':
    main()
