import argparse
import csv
import importlib
import inspect
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import zoom
from tqdm import tqdm

from datasets.acdc_semantic import ACDCVolumeDataset
from training.acdc_evaluation import inference_context


ACDC_CLASS_NAMES = [
    "background",
    "rv",
    "myocardium",
    "lv",
]

ACDC_CLASS_COLORS = [
    "black",
    "red",
    "forestgreen",
    "blue",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize ACDC predictions for EoSeg checkpoints."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--list-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--precision", choices=("16-mixed", "32-true"), default="16-mixed")
    parser.add_argument("--compile-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--roi-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-slices-per-case", type=int, default=0)
    parser.add_argument("--slice-stride", type=int, default=1)
    parser.add_argument("--only-nonempty", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument(
        "--save-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Optional output PNG size. Example: --save-size 224 224",
    )
    return parser.parse_args()


def import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate_from_config(config_node):
    if isinstance(config_node, dict) and "class_path" in config_node:
        cls = import_class(config_node["class_path"])
        init_args = config_node.get("init_args", {})
        kwargs = {k: instantiate_from_config(v) for k, v in init_args.items()}
        sig = inspect.signature(cls.__init__)
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not accepts_var_kwargs:
            valid_keys = {
                name
                for name, p in sig.parameters.items()
                if name != "self"
                and p.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return cls(**kwargs)
    if isinstance(config_node, dict):
        return {k: instantiate_from_config(v) for k, v in config_node.items()}
    if isinstance(config_node, list):
        return [instantiate_from_config(v) for v in config_node]
    return config_node


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_style_links(config: dict, ckpt_path: Path):
    config = yaml.safe_load(yaml.safe_dump(config))

    data_init = config.get("data", {}).get("init_args", {})
    model_init = config.get("model", {}).setdefault("init_args", {})
    network_init = model_init.setdefault("network", {}).setdefault("init_args", {})
    encoder_init = network_init.setdefault("encoder", {}).setdefault("init_args", {})

    if "num_classes" in data_init:
        model_init.setdefault("num_classes", data_init["num_classes"])
        network_init.setdefault("num_classes", data_init["num_classes"])

    if "img_size" in data_init:
        model_init.setdefault("img_size", data_init["img_size"])
        encoder_init.setdefault("img_size", data_init["img_size"])

    encoder_init.setdefault("ckpt_path", str(ckpt_path))
    model_init["ckpt_path"] = None
    return config


def load_model(config: dict, ckpt_path: Path, device: torch.device):
    resolved_config = apply_cli_style_links(config, ckpt_path)
    model_cfg = resolved_config["model"]
    model = instantiate_from_config(model_cfg)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    return model


def normalize_image(image_2d: np.ndarray):
    image_2d = image_2d.astype(np.float32)
    lo = np.percentile(image_2d, 1)
    hi = np.percentile(image_2d, 99)
    if hi <= lo:
        lo = float(image_2d.min())
        hi = float(image_2d.max())
    image_2d = np.clip(image_2d, lo, hi)
    if hi > lo:
        image_2d = (image_2d - lo) / (hi - lo)
    else:
        image_2d = np.zeros_like(image_2d)
    return image_2d


def colorize_mask(mask_2d: np.ndarray):
    rgb = np.zeros((*mask_2d.shape, 3), dtype=np.float32)
    for cls_idx, color_name in enumerate(ACDC_CLASS_COLORS):
        color = np.array(mcolors.to_rgb(color_name), dtype=np.float32)
        rgb[mask_2d == cls_idx] = color
    return rgb


def overlay_mask(image_2d: np.ndarray, mask_2d: np.ndarray, alpha: float = 0.45):
    gray = normalize_image(image_2d)
    base = np.stack([gray, gray, gray], axis=-1)
    color = colorize_mask(mask_2d)
    fg = mask_2d > 0
    base[fg] = (1.0 - alpha) * base[fg] + alpha * color[fg]
    return np.clip(base, 0.0, 1.0)


def resize_for_saving(arr: np.ndarray, save_size, is_mask: bool):
    if save_size is None:
        return arr
    height, width = save_size
    pil = Image.fromarray(arr)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return np.array(pil.resize((width, height), resample=resample))


def infer_volume(model, img_vol_np: np.ndarray, patch_size: tuple[int, int], device: torch.device):
    if (tuple(patch_size) != tuple(model.img_size) or model.device.type != device.type
            or device.index not in (None, model.device.index)):
        raise ValueError("Visualization must use the configured model size and device")
    return model._run_case_inference(img_vol_np)




def select_slice_indices(gt_vol: np.ndarray, pred_vol: np.ndarray, args):
    indices = list(range(0, gt_vol.shape[0], max(args.slice_stride, 1)))
    if args.only_nonempty:
        indices = [
            idx for idx in indices
            if (gt_vol[idx] > 0).any() or (pred_vol[idx] > 0).any()
        ]
    if args.max_slices_per_case > 0 and len(indices) > args.max_slices_per_case:
        picks = np.linspace(0, len(indices) - 1, args.max_slices_per_case, dtype=int)
        indices = [indices[i] for i in picks]
    return indices


def save_case_visualizations(case_dir: Path, case_name: str, image_vol: np.ndarray, gt_vol: np.ndarray, pred_vol: np.ndarray, args):
    slice_indices = select_slice_indices(gt_vol, pred_vol, args)
    legend_handles = []
    for cls_idx in range(1, len(ACDC_CLASS_NAMES)):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=ACDC_CLASS_COLORS[cls_idx],
                lw=4,
                label=f"{cls_idx}: {ACDC_CLASS_NAMES[cls_idx]}",
            )
        )

    for idx in slice_indices:
        image_2d = image_vol[idx]
        gt_2d = gt_vol[idx]
        pred_2d = pred_vol[idx]

        image_gray = normalize_image(image_2d)
        gt_overlay = overlay_mask(image_2d, gt_2d)
        pred_overlay = overlay_mask(image_2d, pred_2d)

        image_to_save = resize_for_saving((image_gray * 255).astype(np.uint8), args.save_size, is_mask=False)
        gt_to_save = resize_for_saving((gt_overlay * 255).astype(np.uint8), args.save_size, is_mask=False)
        pred_to_save = resize_for_saving((pred_overlay * 255).astype(np.uint8), args.save_size, is_mask=False)

        plt.imsave(case_dir / f"{case_name}_slice_{idx:03d}_image.png", image_to_save, cmap="gray")
        plt.imsave(case_dir / f"{case_name}_slice_{idx:03d}_gt.png", gt_to_save)
        plt.imsave(case_dir / f"{case_name}_slice_{idx:03d}_pred.png", pred_to_save)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(image_to_save, cmap="gray")
        axes[0].set_title(f"{case_name} slice {idx:03d} image")
        axes[1].imshow(gt_to_save)
        axes[1].set_title("ground truth")
        axes[2].imshow(pred_to_save)
        axes[2].set_title("prediction")

        for ax in axes:
            ax.axis("off")

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.02),
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout()
        fig.savefig(case_dir / f"{case_name}_slice_{idx:03d}_panel.png", dpi=250, bbox_inches="tight")
        plt.close(fig)




def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("medium")

    config = load_config(args.config)
    model = load_model(config, args.ckpt, device)
    dataset = ACDCVolumeDataset(
        base_dir=str(args.base_dir),
        list_dir=str(args.list_dir),
        split="test",
        ignore_index=config["data"]["init_args"].get("ignore_idx", 255),
    )

    if args.max_cases > 0:
        dataset.cases = dataset.cases[: args.max_cases]

    patch_size = tuple(config["model"]["init_args"]["img_size"])
    metrics_rows = []

    with inference_context(model, precision=args.precision, compile_test=args.compile_test,
                           roi_metrics=args.roi_metrics):
        for sample in tqdm(dataset, total=len(dataset), desc="Visualizing"):
            case_name = sample["case_name"]
            image_vol = sample["image"].numpy().astype(np.float32)
            gt_vol = sample["label"].numpy().astype(np.int32)
            pred_vol = infer_volume(model, image_vol, patch_size, device)
    
            case_dir = args.output_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            save_case_visualizations(case_dir, case_name, image_vol, gt_vol, pred_vol, args)
    
            if args.save_npy:
                np.save(case_dir / f"{case_name}_pred.npy", pred_vol)
    
            metrics_rows.append({"case_name": case_name,
                                 **model._compute_case_metrics(pred_vol, gt_vol)})

    csv_path = args.output_dir / "metrics_summary.csv"
    if metrics_rows:
        fieldnames = list(metrics_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"[done] saved visualizations to {args.output_dir}")
        print(f"[done] saved metrics to {csv_path}")
        print(f"[done] average mean_dice across cases: {np.mean([row['mean_dice'] for row in metrics_rows]):.4f}")
    else:
        print("[done] no cases were processed")


if __name__ == "__main__":
    main()
