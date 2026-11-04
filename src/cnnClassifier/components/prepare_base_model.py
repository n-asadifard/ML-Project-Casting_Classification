from cnnClassifier.constants import *
import tensorflow as tf
import keras, math, os
import pandas as pd
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import WarmupCosine, save_model
from cnnClassifier.entity.config_entity import (PrepareBaseModelConfig, PrepareBaseModelParams)

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig, params: PrepareBaseModelParams):
        self.config = config
        self.params = params

    
    def get_base_model(self):

        self.model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.params.IMG_SIZE, 3),    # the base always sees 3 channels
            weights="imagenet",
            include_top=self.params.INCLUDE_TOP,  # must be False
            )
        logger.info("base model is created")

        save_model(path=self.config.base_model_path, model=self.model)


    def _set_trainable(self, model, freeze_all: bool, n_unfreeze: int | None) -> None:
        """Freeze the whole base, or unfreeze only its top n layers (BatchNorm always stays frozen)."""
        if freeze_all:
            if n_unfreeze:
                raise ValueError("freeze_all=True cannot be combined with n_unfreeze")
            model.trainable = False
            return

        if n_unfreeze is None or n_unfreeze <= 0:
            raise ValueError("Either set freeze_all=True or pass n_unfreeze > 0")
        if n_unfreeze > len(model.layers):
            raise ValueError(f"n_unfreeze={n_unfreeze} exceeds the {len(model.layers)} layers of the base model")

        model.trainable = True
        for layer in model.layers[:-n_unfreeze]:
            layer.trainable = False
        for layer in model.layers[-n_unfreeze:]:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

            
    def _prepare_full_model(self, model, freeze_all: bool, n_unfreeze: int, lr_schedule):
        """Builds and compiles the full model around the base model."""
        self._set_trainable(model, freeze_all, n_unfreeze)
        
        inputs = keras.Input(shape=(*self.params.IMG_SIZE, 1), name="gray_input")
        x = keras.layers.Concatenate(name="gray_to_rgb")([inputs, inputs, inputs])
        x = keras.layers.Rescaling(scale=2.0, offset=-1.0, name="to_minus1_1")(x)
        # x = model(x, training=False)    # keeps BatchNorm in inference mode
        x = model(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(self.params.DROPOUT)(x)
        outputs = keras.layers.Dense(1, activation="sigmoid", name="defect_prob")(x)

        full_model = keras.Model(inputs=inputs, outputs=outputs, name="updated_mobilenetv2")

        full_model.compile(
            jit_compile=False,
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
                keras.metrics.AUC(name="auc"),
                keras.metrics.Precision(name="precision", thresholds=0.5),
                keras.metrics.Recall(name="recall", thresholds=0.5),
            ])
        
        full_model.summary()
        return full_model
    
    
    def create_lr_schedule(self, fine_tune: bool, dataset: str):
        """Return the correct lr_schedule based on the flag 'fine_tune'."""
        target_split=''
        for split in Path(self.config).iterdir():
            if split.name == dataset:
                target_split = str(split)
        n_train = len(pd.read_csv(Path(os.path.join(self.config.manifest_path, target_split))))
        steps_per_epoch = math.ceil(n_train / self.params.BATCH_SIZE)

        if not fine_tune:
            return WarmupCosine(
                peak_lr=self.params.LEARNINGRATE_P1,
                total_steps=self.params.EPOCHS_P1 * steps_per_epoch,
                warmup_steps=2 * steps_per_epoch,
            )
        return WarmupCosine(
            peak_lr=self.params.LEARNINGRATE_P2,
            total_steps=self.params.EPOCHS_P2 * steps_per_epoch,
            warmup_steps=steps_per_epoch,
            final_lr=1e-7,
        )
    
    
    def update_base_model(self, freeze_all, n_unfreeze, fine_tune, init_weights_path=None):
        if fine_tune == freeze_all:
            raise ValueError("fine_tune and freeze_all contradict each other: "
                             "phase 1 = freeze_all=True, fine_tune=False; "
                             "phase 2 = freeze_all=False, fine_tune=True")

        self.full_model = self._prepare_full_model(
            model=self.model,
            freeze_all=freeze_all,
            n_unfreeze=n_unfreeze,
            lr_schedule=self.create_lr_schedule(fine_tune),
        )
        if init_weights_path is not None:
            self.full_model.load_weights(init_weights_path)   # head from phase 1, fresh optimizer
        
        logger.info("updated base model is created")
        save_model(path=self.config.updated_base_model_path, model=self.full_model)