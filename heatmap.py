import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

import config
import dataset
import model_utils

torch.backends.cudnn.enabled = False


def _compute_heatmap(pil_image, model, device, transform):
    tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_id = probs.argmax().item()

    attentions = outputs.attentions[config.ATTENTION_LAYER_INDEX]
    att = attentions[0].mean(dim=0)
    cls_att = att[0, 1:].cpu().numpy()

    num_patches_side = config.IMAGE_SIZE // config.PATCH_SIZE
    att_map = cls_att.reshape(num_patches_side, num_patches_side)
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
    att_resized = np.array(
        Image.fromarray((att_map * 255).astype(np.uint8)).resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR
        )
    ) / 255.0

    img_display = pil_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))

    return img_display, att_resized, config.ID2LABEL[pred_id], probs[pred_id].item()


def generate_dataset_grid(n=9, checkpoint_path=None):
    device = torch.device(config.DEVICE)
    model = model_utils.load_trained_model(checkpoint_path)
    transform = dataset.get_val_transforms()

    print(f"Cargando dataset desde el Hub: {config.DATASET_NAME}...")
    raw_dataset = load_dataset(config.DATASET_NAME)
    split = raw_dataset.get("test") or raw_dataset["train"]
    indices = random.sample(range(len(split)), min(n, len(split)))
    print(f"Generando heatmaps para {len(indices)} imágenes aleatorias...")

    # Cada muestra ocupa 2 columnas (original + heatmap), con máximo 3 muestras por fila
    samples_per_row = 3
    rows = (len(indices) + samples_per_row - 1) // samples_per_row
    cols = min(samples_per_row, len(indices)) * 2

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = np.array(axes).reshape(rows, cols)

    for i, idx in enumerate(indices):
        row = i // samples_per_row
        col = (i % samples_per_row) * 2

        pil_image = split[idx]["image"].convert("RGB")
        real_label = config.ID2LABEL.get(split[idx].get("label", -1), "?")
        img_display, att_resized, pred_label, confidence = _compute_heatmap(
            pil_image, model, device, transform
        )

        # Original
        axes[row, col].imshow(img_display)
        axes[row, col].set_title(f"Real: {real_label}", fontsize=8)
        axes[row, col].axis("off")

        # Heatmap
        img_gray = np.array(img_display.convert("L"))
        axes[row, col + 1].imshow(img_gray, cmap="gray")
        axes[row, col + 1].imshow(att_resized, cmap="jet", alpha=config.HEATMAP_ALPHA)
        axes[row, col + 1].set_title(f"Pred: {pred_label} ({confidence:.1%})", fontsize=8)
        axes[row, col + 1].axis("off")

    # Ocultar ejes sobrantes si n no rellena la última fila
    for i in range(len(indices), rows * samples_per_row):
        row = i // samples_per_row
        col = (i % samples_per_row) * 2
        axes[row, col].axis("off")
        axes[row, col + 1].axis("off")

    fig.suptitle("Attention Maps — ViT Mama", fontsize=13, y=1.01)
    plt.tight_layout()

    output_dir = os.path.join(config.RESULTS_DIR, "heatmaps")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "grid.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Grid guardado en '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=9,
                        help="Número de imágenes aleatorias del dataset (default: 9)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Ruta al checkpoint (default: checkpoints/best_model.pth)")
    args = parser.parse_args()

    generate_dataset_grid(args.n, args.checkpoint)
