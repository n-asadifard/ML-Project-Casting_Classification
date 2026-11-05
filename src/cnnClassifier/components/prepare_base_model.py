from cnnClassifier.constants import *
from cnnClassifier import logger
from cnnClassifier.utils.common import set_trainable, save_model
from cnnClassifier.entity.config_entity import (PrepareBaseModelConfig, PrepareBaseModelParams)
import tensorflow as tf
import keras

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig, params: PrepareBaseModelParams):
        self.config = config
        self.params = params

    def get_base_model(self):
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.params.IMG_SIZE, 3),   # the base always sees 3 channels
            weights="imagenet",
            include_top=self.params.INCLUDE_TOP,      # must be False
        )
        logger.info("base model is created")
        save_model(path=self.config.base_model_path, model=self.model)

    def update_base_model(self):
        """Wrap the frozen base in grayscale input + head. Saved UNCOMPILED: compiling is phase-specific (04)."""
        set_trainable(self.model, freeze_all=True)

        inputs = keras.Input(shape=(*self.params.IMG_SIZE, 1), name="gray_input")
        x = keras.layers.Concatenate(name="gray_to_rgb")([inputs, inputs, inputs])
        x = keras.layers.Rescaling(scale=2.0, offset=-1.0, name="to_minus1_1")(x)
        x = self.model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(self.params.DROPOUT)(x)
        outputs = keras.layers.Dense(1, activation="sigmoid", name="defect_prob")(x)

        self.full_model = keras.Model(inputs, outputs, name="updated_mobilenetv2")
        self.full_model.summary()
        logger.info("updated base model is created")
        save_model(path=self.config.updated_base_model_path, model=self.full_model)