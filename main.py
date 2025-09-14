import torch
import custom_data_processing
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from Models import HandPoseVGG16
import time
import os


def _ensure_targets_shape_dtype(t):
    t = t.to(DEVICE).float()
    if t.ndim == 3 and t.shape[1:] == (21, 3):
        t = t.view(t.size(0), -1)  # -> [B,63]
    return t


def evaluate(loader):
    model.eval()
    mse_sum, n_batches = 0.0, 0
    with torch.no_grad():
        for ims, targets in loader:
            ims = ims.to(DEVICE)
            targets = _ensure_targets_shape_dtype(targets)
            preds = model(ims)
            mse = nn.functional.mse_loss(preds, targets, reduction="mean")
            mse_sum += mse.item()
            n_batches += 1
    return mse_sum / max(n_batches, 1)

JOINT_LIST_NYU = [31, 28, 23, 17, 11, 5, 27, 22, 16, 10, 4, 25, 20, 14, 8, 2, 24, 18, 12, 6, 0]
ROOT = "/Volumes/Evan_Samsung_HP_data/nyu_dataset/data"
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_PATH  = os.path.join(SAVE_DIR, "handpose_vgg16_best.pt")
FINAL_PATH = os.path.join(SAVE_DIR, "handpose_vgg16_final.pt")
PRED_PATH  = os.path.join(SAVE_DIR, "predictions.npy")     # [N, 63]
GT_PATH    = os.path.join(SAVE_DIR, "ground_truths.npy")   # [N, 63]
TEST_PRED_PATH = os.path.join(SAVE_DIR, "test_predictions.npy")   # [N_test, 63]
TEST_GT_PATH   = os.path.join(SAVE_DIR, "test_ground_truths.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
VAL_FRACTION = 0.2
BATCH_SIZE = 32
PATIENCE = 5
SEED = 42

# For debugging purposes only so we get the same results each run if necessary
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# data loading
full_dataset = custom_data_processing.HandposeDataset(ROOT, JOINT_LIST_NYU)
test_dataset = custom_data_processing.HandposeDataset(ROOT, JOINT_LIST_NYU, mode="test")
val_len = int(len(full_dataset)*VAL_FRACTION)
train_len = len(full_dataset) - val_len
train_ds, val_ds = random_split(full_dataset, lengths=[train_len, val_len],
                                generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0,
                          pin_memory=(DEVICE.type == "cuda"))
val_loader = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=0,
                        pin_memory=(DEVICE.type == "cuda"))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0,
                        pin_memory=(DEVICE.type == "cuda"))


model = HandPoseVGG16(out_dim=63).to(DEVICE)
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()
best_val_loss = float("inf")
epochs_no_improve = 0
start = time.time()
for epoch in range(1, EPOCHS + 1):
    print("check it out")
    print(f"{time.time() - start} seconds")
    start = time.time()
    model.train()
    run_loss, n_batches = 0.0, 0
    for ims, target in train_loader:
        ims = ims.to(DEVICE)
        targets = _ensure_targets_shape_dtype(target)

        preds = model(ims)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run_loss += loss.item()
        n_batches += 1
    train_mse = run_loss / max(n_batches, 1)
    val_mse = evaluate(val_loader)
    print(f"Epoch {epoch}/{EPOCHS}  |  train MSE: {train_mse:.6f}  |  val MSE: {val_mse:.6f}")


    # Save best-on-validation
    if val_mse < best_val_loss - 1e-8:  # tiny epsilon to avoid float jitter
        best_val_loss = val_mse
        epochs_no_improve = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }, BEST_PATH)
        print(f"  ✔ New best (val MSE {best_val_loss:.6f}). Saved to {BEST_PATH}")
    else:
        epochs_no_improve += 1
        print(f"  ↳ No improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered (patience={PATIENCE}).")
            break

# Save final (last-epoch) weights
torch.save(model.state_dict(), FINAL_PATH)
print(f"Saved final model weights to {FINAL_PATH}")


# --------------------
# Evaluation on validation split + save preds/GT (optional)
# --------------------
model.eval()
all_preds, all_targets = [], []
mse_sum, mae_sum, n_batches = 0.0, 0.0, 0

with torch.no_grad():
    for ims, targets in test_loader:
        ims = ims.to(DEVICE)
        targets = _ensure_targets_shape_dtype(targets)  # [B,63]
        preds = model(ims)                               # [B,63]

        # metrics
        mse = nn.functional.mse_loss(preds, targets, reduction="mean")
        mae = nn.functional.l1_loss(preds, targets, reduction="mean")
        mse_sum += mse.item()
        mae_sum += mae.item()
        n_batches += 1

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

test_mse = mse_sum / max(n_batches, 1)
test_mae = mae_sum / max(n_batches, 1)
print(f"[TEST] MSE: {test_mse:.6f} | MAE: {test_mae:.6f}")

# Save test predictions/GT
if len(all_preds) > 0:
    all_preds = np.concatenate(all_preds, axis=0)     # [N_test, 63]
    all_targets = np.concatenate(all_targets, axis=0) # [N_test, 63]
    np.save(TEST_PRED_PATH, all_preds)
    np.save(TEST_GT_PATH, all_targets)
    print(f"Saved test predictions to {TEST_PRED_PATH} and ground truths to {TEST_GT_PATH}")

