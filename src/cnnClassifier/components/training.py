import keras, math
import pandas as pd
from cnnClassifier.constants import *
from cnnClassifier import logger
from cnnClassifier.utils.common import make_dataset, compute_class_weights, WarmupCosine, save_model, set_trainable
from cnnClassifier.entity.config_entity import TrainingConfig, TrainingParams


class Training:
    def __init__(self, config: TrainingConfig, params: TrainingParams):
        self.config = config
        self.params = params
        self.img_size = tuple(params.IMG_SIZE)

    # ---------- model ----------
    def get_base_model(self):
        """Load the architecture from 02 (uncompiled) and locate the nested MobileNetV2."""
        self.model = keras.models.load_model(self.config.updated_base_model_path, compile=False)
        self.base_model = next(l for l in self.model.layers if isinstance(l, keras.Model))

    # ---------- data ----------
    def build_augmenter(self, seed: int = 42) -> keras.Sequential:
        return keras.Sequential([
            keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            keras.layers.RandomRotation(self.params.ROTATION_FACTOR, fill_mode="reflect", seed=seed),
            keras.layers.RandomBrightness(self.params.BRIGHTNESS_FACTOR, value_range=(0.0, 1.0), seed=seed),
            keras.layers.RandomContrast(self.params.CONTRAST_FACTOR, seed=seed),
        ], name="augmenter")
    

    def build_datasets(self):
        """Train (augmented) and val only. The test split is reserved for the evaluation stage."""
        splits = Path(self.config.splits_dir)
        train_df = pd.read_csv(splits / "train.csv")
        self.class_weights = compute_class_weights(train_df)
        self.steps_per_epoch = math.ceil(len(train_df) / self.params.BATCH_SIZE)

        self.train_ds = make_dataset(splits / "train.csv", self.img_size, self.params.BATCH_SIZE,
                                    training=True, augmenter=self.build_augmenter())
        self.val_ds = make_dataset(splits / "val.csv", self.img_size, self.params.BATCH_SIZE)
        logger.info(f"datasets ready | steps/epoch={self.steps_per_epoch} | class weights={self.class_weights}")
        

    # ---------- phase-specific training setup ----------
    def _create_lr_schedule(self, phase: int) -> WarmupCosine:
        p = self.params
        if phase == 1:
            epochs, peak, warmup_epochs, final_lr = p.EPOCHS_P1, p.LEARNINGRATE_P1, p.WARMUP_EPOCHS_P1, 1e-6
        else:
            epochs, peak, warmup_epochs, final_lr = p.EPOCHS_P2, p.LEARNINGRATE_P2, p.WARMUP_EPOCHS_P2, 1e-7
        return WarmupCosine(peak_lr=peak,
                            total_steps=epochs * self.steps_per_epoch,
                            warmup_steps=warmup_epochs * self.steps_per_epoch,
                            final_lr=final_lr)

    def _compile(self, lr_schedule) -> None:
        self.model.compile(
            jit_compile=False,
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),   # fresh optimizer per phase
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
                keras.metrics.AUC(name="auc"),
                keras.metrics.Precision(name="precision", thresholds=0.5),
                keras.metrics.Recall(name="recall", thresholds=0.5),
            ],
        )

    def run_phase(self, phase: int, callback_list: list):
        """Phase 1: train the head, base frozen. Phase 2: fine-tune the top layers from the phase-1 weights."""
        if phase not in (1, 2):
            raise ValueError("phase must be 1 or 2")

        if phase == 1:
            set_trainable(self.base_model, freeze_all=True)
            epochs = self.params.EPOCHS_P1
            best_weights = self.config.phase1_weights_path
        else:
            # start from the best phase-1 head, also after a kernel restart
            self.model.load_weights(self.config.phase1_weights_path)
            set_trainable(self.base_model, freeze_all=False, n_unfreeze=self.params.N_UNFREEZE)
            epochs = self.params.EPOCHS_P2
            best_weights = self.config.phase2_weights_path

        self._compile(self._create_lr_schedule(phase))   # required after changing trainable flags
        self.model.summary()

        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            class_weight=self.class_weights,
            callbacks=callback_list,
        )

        # the checkpoint file is the source of truth for "best epoch" of this phase
        self.model.load_weights(best_weights)
        logger.info(f"phase {phase} finished, best weights reloaded from {best_weights}")
        return history

    def save_trained_model(self):
        """Full model (architecture + weights) for evaluation and inference."""
        save_model(path=self.config.trained_model_path, model=self.model)
        logger.info(f"trained model saved to {self.config.trained_model_path}")