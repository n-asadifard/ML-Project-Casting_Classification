import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             average_precision_score, accuracy_score, precision_score,
                             recall_score, f1_score, fbeta_score)
from cnnClassifier import logger
from cnnClassifier.constants import *
from cnnClassifier.utils.common import (save_json, make_dataset, load_image)
from cnnClassifier.entity.config_entity import EvaluationParams, EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig, params: EvaluationParams):
        self.config = config
        self.params = params
        self.img_size = tuple(params.IMG_SIZE)
        self.class_names = sorted(params.CLASS_TO_IDX, key=params.CLASS_TO_IDX.get)   # index order

    def load_model(self):
        # compile=False: we only predict, so the optimizer/schedule from training is not needed
        self.model = keras.models.load_model(self.config.trained_model_path, compile=False)

    def predict_split(self, split: str):
        """P(defect) for every image of a split, in manifest order (val/test are not shuffled)."""
        manifest = Path(self.config.splits_dir) / f"{split}.csv"
        df = pd.read_csv(manifest)
        ds = make_dataset(manifest, self.img_size, self.params.BATCH_SIZE)   # training=False
        y_prob = self.model.predict(ds, verbose=0).ravel()
        if len(y_prob) != len(df):
            raise RuntimeError(f"{split}: {len(y_prob)} predictions for {len(df)} manifest rows")
        return df, df["label"].to_numpy(), y_prob

    def choose_threshold(self, y_true, y_prob, grid=np.linspace(0.05, 0.95, 91)):
        """Threshold maximizing F-beta; on ties, the middle of the best plateau."""
        scores = np.array([fbeta_score(y_true, y_prob >= t, beta=self.params.THRESHOLD_BETA,
                                       zero_division=0) for t in grid])
        best = grid[scores >= scores.max() - 1e-9]
        return float(np.median(best)), float(scores.max()), (float(best.min()), float(best.max()))

    def evaluate_split(self, name: str, y_true, y_prob, threshold: float) -> dict:
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        print(f"===== {name} | threshold = {threshold:.3f} =====")
        print("Confusion matrix (rows = true, cols = pred; order: ok, defect)")
        print(cm)
        print(classification_report(y_true, y_pred, target_names=self.class_names, digits=3, zero_division=0))
        metrics = {
            "threshold": float(threshold),
            "n": int(len(y_true)),
            "confusion_matrix": cm.tolist(),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_defect": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall_defect": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_defect": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
        }
        print(f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}\n")
        return metrics

    def select_threshold(self, y_true, y_prob):
        """Validation only: show the default 0.5, then pick and report the F-beta threshold."""
        self.evaluate_split("VAL @ default 0.5", y_true, y_prob, 0.5)
        thr, best, (lo, hi) = self.choose_threshold(y_true, y_prob)
        logger.info(f"threshold {thr:.3f} chosen on val (F{self.params.THRESHOLD_BETA:g}={best:.3f}, plateau {lo:.2f} to {hi:.2f})")
        return thr, self.evaluate_split("VAL @ chosen threshold", y_true, y_prob, thr)

    def show_misclassified(self, df, y_prob, threshold: float, max_n: int = 8) -> None:
        pred = (y_prob >= threshold).astype(int)
        wrong = df.assign(prob=y_prob, pred=pred)[pred != df["label"].to_numpy()].head(max_n)
        print(f"{int((pred != df['label'].to_numpy()).sum())} misclassified of {len(df)}")
        if wrong.empty:
            return
        fig, axes = plt.subplots(1, len(wrong), figsize=(3.2 * len(wrong), 3.6), squeeze=False)
        for ax, (_, r) in zip(axes[0], wrong.iterrows()):
            img = load_image(tf.constant(r["filepath"]), self.img_size)
            ax.imshow(img[..., 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"true: {self.class_names[int(r['label'])]}\nP(defect)={r['prob']:.2f}", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def save_report(self, threshold: float, val_metrics: dict, test_metrics: dict) -> None:
        save_json(self.config.report_path, {"threshold": threshold, "val": val_metrics, "test": test_metrics})

    def save_metadata(self, threshold: float) -> None:
        """Everything the predictor needs, so nothing is hard-coded at inference time."""
        save_json(self.config.metadata_path, {
            "model_path": str(self.config.trained_model_path),
            "class_to_idx": dict(self.params.CLASS_TO_IDX),
            "threshold": float(threshold),
            "input": {"size": list(self.img_size), "channels": 1, "range": "[0, 1] grayscale"},
            "in_model_preprocessing": "gray->RGB (3x concat), then Rescaling to [-1, 1]",
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
        })