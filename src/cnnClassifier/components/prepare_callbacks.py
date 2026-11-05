from pathlib import Path
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    def _checkpoint_path(self, phase: int) -> Path:
        return {1: self.config.checkpoint_phase1_filepath,
                2: self.config.checkpoint_phase2_filepath}[phase]

    def get_callbacks(self, phase: int, patience: int = 5) -> list:
        if phase not in (1, 2):
            raise ValueError("phase must be 1 or 2")
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_dir = Path(self.config.tensorboard_root_log_dir) / f"phase{phase}" / f"tb_logs_at_{timestamp}"
        return [
            tf.keras.callbacks.TensorBoard(log_dir=str(tb_dir)),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self._checkpoint_path(phase)),
                monitor="val_loss", mode="min",
                save_best_only=True, save_weights_only=True),   # the phase handoff file
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min",
                patience=patience, restore_best_weights=True),
        ]