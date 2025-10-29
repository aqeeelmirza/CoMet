# ------------------------------------------------------------------
# CoMet: https://github.com/aqeeelmirza/CoMet
# ------------------------------------------------------------------

import csv
import os
import random
from typing import List, Optional, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
import tqdm


def plot_segmentation_images(
    save_folder: str,
    image_paths: List[str],
    segmentations: List[np.ndarray],
    anomaly_scores: Optional[List[float]] = None,
    mask_paths: Optional[List[str]] = None,
    image_transform: Callable = lambda x: x,
    mask_transform: Callable = lambda x: x,
    save_depth: int = 4,
) -> None:
    """
    Save visualization of input image, ground truth mask (if provided), and predicted segmentation.

    Args:
        save_folder: Directory to save images.
        image_paths: List of input image paths.
        segmentations: List of predicted anomaly maps (H, W).
        anomaly_scores: Optional list of anomaly scores.
        mask_paths: Optional list of ground truth mask paths.
        image_transform: Transform to apply to input image.
        mask_transform: Transform to apply to mask.
        save_depth: How many path segments to include in filename.
    """
    os.makedirs(save_folder, exist_ok=True)
    if mask_paths is None:
        mask_paths = [None] * len(image_paths)
    if anomaly_scores is None:
        anomaly_scores = [None] * len(image_paths)

    for img_path, mask_path, score, seg in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Saving visualizations",
        leave=False,
    ):
        # Load and transform image
        img = PIL.Image.open(img_path).convert("RGB")
        img = image_transform(img)
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()

        # Load mask if provided
        if mask_path and os.path.exists(mask_path):
            mask = PIL.Image.open(mask_path).convert("L")
            mask = mask_transform(mask)
            if isinstance(mask, torch.Tensor):
                mask = mask.squeeze(0).cpu().numpy()
            mask = np.stack([mask] * 3, axis=-1)  # to RGB
        else:
            mask = np.zeros_like(img)

        # Prepare filename
        parts = img_path.split(os.sep)
        savename = "_".join(parts[-save_depth:]).replace(".png", ".jpg")
        savepath = os.path.join(save_folder, savename)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(img)
        axes[0].set_title(f"Image\nScore: {score:.3f}" if score is not None else "Image")
        axes[0].axis("off")

        axes[1].imshow(mask)
        axes[1].set_title("GT Mask")
        axes[1].axis("off")

        im = axes[2].imshow(seg, cmap="jet", vmin=0, vmax=1)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        plt.tight_layout()
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close(fig)


def create_storage_folder(
    root: str,
    project: str,
    group: str,
    run_name: str,
    mode: str = "iterate",
) -> str:
    """
    Create a unique run directory.

    Args:
        root: Root results directory.
        project: Project name.
        group: Experiment group.
        run_name: Run identifier.
        mode: "iterate" (auto-increment) or "overwrite".

    Returns:
        Full path to created folder.
    """
    os.makedirs(root, exist_ok=True)
    project_path = os.path.join(root, project)
    os.makedirs(project_path, exist_ok=True)

    save_path = os.path.join(project_path, group, run_name)
    if mode == "iterate":
        base_path = os.path.join(project_path, group)
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(base_path + f"_{counter}")
            counter += 1
    os.makedirs(save_path, exist_ok=True)
    return save_path


def set_torch_device(gpu_ids: List[int]) -> torch.device:
    """Select GPU or CPU device."""
    if gpu_ids and torch.cuda.is_available():
        device_id = gpu_ids[0] % torch.cuda.device_count()
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def fix_seeds(seed: int, torch_seed: bool = True, cuda: bool = True) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch_seed:
        torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_and_store_final_results(
    results_path: str,
    results: List[List[float]],
    row_names: Optional[List[str]] = None,
    column_names: Optional[List[str]] = None,
) -> dict:
    """
    Save per-dataset results and compute mean metrics.

    Args:
        results_path: Directory to save CSV.
        results: List of [instance_auroc, full_pixel_auroc, anomaly_pixel_auroc]
        row_names: Dataset names.
        column_names: Metric names.

    Returns:
        Dictionary of mean metrics.
    """
    if column_names is None:
        column_names = [
            "Instance AUROC",
            "Full Pixel AUROC",
            "Anomaly Pixel AUROC",
        ]

    if row_names is not None:
        assert len(row_names) == len(results), "Row names must match results length."

    # Compute mean
    mean_values = np.mean(results, axis=0)
    mean_metrics = {f"mean_{name.lower().replace(' ', '_')}": val for name, val in zip(column_names, mean_values)}

    # Save CSV
    csv_path = os.path.join(results_path, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = column_names
        if row_names is not None:
            header = ["Dataset"] + header
        writer.writerow(header)

        for i, row in enumerate(results):
            csv_row = row
            if row_names is not None:
                csv_row = [row_names[i]] + row
            writer.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in csv_row])

        mean_row = ["Mean"] + [f"{x:.4f}" for x in mean_values]
        writer.writerow(mean_row)

    return mean_metrics
