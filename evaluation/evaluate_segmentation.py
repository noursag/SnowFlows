import numpy as np
import rasterio
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rgb_dir = os.path.join(BASE_DIR, "data", "rgb")
mask_dir = os.path.join(BASE_DIR, "data", "masks")

# -------------------------
# LOAD TIFF
# -------------------------
def load_tif(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)

# -------------------------
# REAL NDSI PREDICTION
# -------------------------
def compute_ndsi(img):
    """
    Expecting:
    band order must include B3 and B11
    """

    if img.shape[0] < 2:
        raise ValueError("Not enough bands for NDSI")

    green = img[1]   # B3
    swir  = img[-1]  # B11 (last band)

    ndsi = (green - swir) / (green + swir + 1e-8)

    return (ndsi > 0.4).astype(np.uint8)

# -------------------------
# METRICS
# -------------------------
def metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection

    iou = intersection / (union + 1e-8)
    dice = (2 * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)
    acc = np.mean(pred == gt)

    return iou, dice, acc

# -------------------------
# EVALUATION LOOP
# -------------------------
ious, dices, accs = [], [], []

mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])

for f in mask_files:

    mask_path = os.path.join(mask_dir, f)
    rgb_path = os.path.join(rgb_dir, f.replace("mask", "rgb"))

    gt = load_tif(mask_path)[0].astype(np.uint8)
    img = load_tif(rgb_path)

    pred = compute_ndsi(img)

    iou, dice, acc = metrics(pred, gt)

    ious.append(iou)
    dices.append(dice)
    accs.append(acc)

print("===================================")
print("REAL CV SEGMENTATION EVALUATION")
print("===================================")
print("IoU:", np.mean(ious))
print("Dice:", np.mean(dices))
print("Accuracy:", np.mean(accs))
print(img.shape)