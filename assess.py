import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from scipy.signal import convolve2d
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from TU2Net import Generator_full


class Get_tager_sample(Dataset):
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.path}")
        self.img_path = sorted(
            item.name for item in self.path.iterdir() if item.suffix.lower() == ".npy"
        )
        if not self.img_path:
            raise ValueError(f"No .npy samples were found in {self.path}.")

    def __getitem__(self, idx: int):
        img_name = self.img_path[idx]
        array = np.load(self.path / img_name)
        tensor = torch.as_tensor(array, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 4 or tensor.shape[0] < 10 or tensor.shape[1] != 1:
            raise ValueError(
                f"{img_name} must have shape [T, 1, H, W] or [T, H, W] with T >= 10."
            )
        satellite = tensor[:10]
        minimum = float(satellite.min())
        maximum = float(satellite.max())
        if minimum < -1e-5 or maximum > 1.00001:
            raise ValueError(
                f"{img_name} satellite values are outside [0, 1]: min={minimum}, max={maximum}."
            )
        wind = (
            tensor[10:14]
            if tensor.shape[0] >= 14
            else torch.empty((0, 1, tensor.shape[-2], tensor.shape[-1]))
        )
        return tensor[:4], tensor[4:10], wind, img_name

    def __len__(self) -> int:
        return len(self.img_path)


def fraction_skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    window_size: int = 5,
) -> float:
    true_binary = (np.asarray(y_true) >= threshold).astype(np.float64)
    pred_binary = (np.asarray(y_pred) >= threshold).astype(np.float64)
    kernel = np.ones((window_size, window_size), dtype=np.float64) / float(
        window_size * window_size
    )
    true_fraction = convolve2d(true_binary, kernel, mode="same", boundary="fill")
    pred_fraction = convolve2d(pred_binary, kernel, mode="same", boundary="fill")
    numerator = np.mean((true_fraction - pred_fraction) ** 2)
    denominator = np.mean(true_fraction**2) + np.mean(pred_fraction**2)
    return 1.0 if denominator == 0.0 else float(1.0 - numerator / denominator)


def neighborhood_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    radius: float = 1.0,
) -> tuple[int, int, int, int]:
    true_binary = np.asarray(y_true) >= threshold
    pred_binary = np.asarray(y_pred) >= threshold
    reference_coordinates = np.argwhere(true_binary)
    prediction_coordinates = np.argwhere(pred_binary)
    hits = 0
    if len(reference_coordinates) and len(prediction_coordinates):
        if radius == 0:
            reference_set = {tuple(coordinate) for coordinate in reference_coordinates}
            hits = sum(tuple(coordinate) in reference_set for coordinate in prediction_coordinates)
        else:
            tree = cKDTree(reference_coordinates)
            used = np.zeros(len(reference_coordinates), dtype=bool)
            for prediction in prediction_coordinates:
                candidates = tree.query_ball_point(prediction, r=radius)
                if not candidates:
                    continue
                ordered = sorted(
                    candidates,
                    key=lambda index: (
                        float(np.sum((reference_coordinates[index] - prediction) ** 2)),
                        index,
                    ),
                )
                for candidate in ordered:
                    if not used[candidate]:
                        used[candidate] = True
                        hits += 1
                        break
    false_alarms = int(len(prediction_coordinates) - hits)
    misses = int(len(reference_coordinates) - hits)
    total = int(true_binary.size)
    correct_negatives = max(total - hits - false_alarms - misses, 0)
    return int(hits), false_alarms, misses, correct_negatives


def metrics_from_counts(
    hits: int, false_alarms: int, misses: int, correct_negatives: int
) -> dict[str, float]:
    total = hits + false_alarms + misses + correct_negatives
    acc = np.nan if total == 0 else (hits + correct_negatives) / total
    csi_denominator = hits + false_alarms + misses
    csi = np.nan if csi_denominator == 0 else hits / csi_denominator
    hss_denominator = (hits + misses) * (misses + correct_negatives) + (
        hits + false_alarms
    ) * (false_alarms + correct_negatives)
    hss = (
        np.nan
        if hss_denominator == 0
        else 2.0
        * (hits * correct_negatives - false_alarms * misses)
        / hss_denominator
    )
    f1_denominator = 2 * hits + false_alarms + misses
    f1 = np.nan if f1_denominator == 0 else 2.0 * hits / f1_denominator
    recall_denominator = hits + misses
    specificity_denominator = correct_negatives + false_alarms
    recall = np.nan if recall_denominator == 0 else hits / recall_denominator
    specificity = (
        np.nan
        if specificity_denominator == 0
        else correct_negatives / specificity_denominator
    )
    bacc = np.nan if np.isnan(recall) or np.isnan(specificity) else (recall + specificity) / 2.0
    return {"ACC": acc, "CSI": csi, "HSS": hss, "F1": f1, "bACC": bacc}


def accuracy(y_true, y_pred, event_threshold, radius=1.0):
    return metrics_from_counts(
        *neighborhood_confusion_counts(y_true, y_pred, event_threshold, radius)
    )["ACC"]


def critical_success_index(y_true, y_pred, event_threshold, radius=1.0):
    return metrics_from_counts(
        *neighborhood_confusion_counts(y_true, y_pred, event_threshold, radius)
    )["CSI"]


def heidke_skill_score(y_true, y_pred, event_threshold, radius=1.0):
    return metrics_from_counts(
        *neighborhood_confusion_counts(y_true, y_pred, event_threshold, radius)
    )["HSS"]


def calculate_f1_score(y_true, y_pred, threshold, radius=1.0):
    return metrics_from_counts(
        *neighborhood_confusion_counts(y_true, y_pred, threshold, radius)
    )["F1"]


def balanced_accuracy(y_true, y_pred, threshold, radius=1.0):
    return metrics_from_counts(
        *neighborhood_confusion_counts(y_true, y_pred, threshold, radius)
    )["bACC"]


def extract_date(filename: str, date_regex: str | None) -> str:
    if not date_regex:
        return ""
    match = re.search(date_regex, filename)
    if not match:
        return ""
    if "date" in match.groupdict():
        return match.group("date")
    if match.groups():
        return match.group(1)
    return match.group(0)


def calculate_metrics(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.2,
    radius: float = 1.0,
    fss_window: int = 5,
    model_variant: str = "complete",
    date_regex: str | None = r"(\d{8})",
    return_rows: bool = False,
):
    metric_names = ("ACC", "CSI", "FSS", "HSS", "F1", "bACC")
    time_step_metrics = None
    overall_metrics = {metric: [] for metric in metric_names}
    rows = []
    model.eval()
    with torch.no_grad():
        for satellite, target, wind, filenames in dataloader:
            satellite = satellite.to(device)
            target = target.to(device)
            wind = wind.to(device)
            if model_variant == "complete":
                if wind.shape[1] != 4:
                    raise ValueError("The complete model requires four DMWC u/v channels.")
                prediction = model(satellite, wind)
            else:
                prediction = model(satellite)
            prediction = torch.clamp(prediction, 0.0, 1.0).cpu().numpy()
            target_array = target.cpu().numpy()
            if prediction.shape != target_array.shape:
                raise ValueError(
                    f"Prediction and target shapes differ: {prediction.shape} versus {target_array.shape}."
                )
            if time_step_metrics is None:
                time_step_metrics = {
                    metric: [[] for _ in range(target_array.shape[1])]
                    for metric in metric_names
                }
            for batch_index, filename in enumerate(filenames):
                for time_index in range(target_array.shape[1]):
                    observed = target_array[batch_index, time_index, 0]
                    predicted = prediction[batch_index, time_index, 0]
                    counts = neighborhood_confusion_counts(
                        observed, predicted, threshold, radius
                    )
                    values = metrics_from_counts(*counts)
                    values["FSS"] = fraction_skill_score(
                        observed, predicted, threshold, fss_window
                    )
                    row = {
                        "sample": filename,
                        "date": extract_date(filename, date_regex),
                        "lead_minutes": (time_index + 1) * 10,
                    }
                    for metric in metric_names:
                        value = float(values[metric])
                        time_step_metrics[metric][time_index].append(value)
                        overall_metrics[metric].append(value)
                        row[metric] = value
                    rows.append(row)
    if time_step_metrics is None:
        raise ValueError("Evaluation loader produced no samples.")
    avg_time_step_metrics = {
        metric: [float(np.nanmean(values)) for values in buckets]
        for metric, buckets in time_step_metrics.items()
    }
    avg_overall_metrics = {
        metric: float(np.nanmean(values)) for metric, values in overall_metrics.items()
    }
    if return_rows:
        return avg_time_step_metrics, avg_overall_metrics, rows
    return avg_time_step_metrics, avg_overall_metrics


def load_generator(
    checkpoint_path: str, device: torch.device, model_variant: str
) -> torch.nn.Module:
    complete = model_variant == "complete"
    model = Generator_full(
        frames=6,
        device=str(device),
        use_lcem=complete,
        use_mfilter=complete,
        use_spade=complete,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["gen"] if isinstance(checkpoint, dict) and "gen" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def write_rows(path: str, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "date",
        "lead_minutes",
        "ACC",
        "CSI",
        "FSS",
        "HSS",
        "F1",
        "bACC",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-variant", choices=("complete", "tu2net"), default="complete")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--fss-window", type=int, default=5)
    parser.add_argument("--date-regex", default=r"(\d{8})")
    parser.add_argument("--output-csv", default="evaluation/per_sequence_metrics.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = Get_tager_sample(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    model = load_generator(args.checkpoint, device, args.model_variant)
    time_step_metrics, overall_metrics, rows = calculate_metrics(
        dataloader,
        model,
        device,
        threshold=args.threshold,
        radius=args.radius,
        fss_window=args.fss_window,
        model_variant=args.model_variant,
        date_regex=args.date_regex,
        return_rows=True,
    )
    write_rows(args.output_csv, rows)
    for time_index in range(len(next(iter(time_step_metrics.values())))):
        print(f"Lead time {(time_index + 1) * 10} min")
        for metric in ("ACC", "CSI", "FSS", "HSS", "F1", "bACC"):
            print(f"{metric}: {time_step_metrics[metric][time_index]:.6f}")
    print("Overall")
    for metric in ("ACC", "CSI", "FSS", "HSS", "F1", "bACC"):
        print(f"{metric}: {overall_metrics[metric]:.6f}")


if __name__ == "__main__":
    main()
