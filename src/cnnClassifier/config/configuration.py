import time
import tensorflow as tf
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (
                                                KaggleToken,
                                                ImageData,
                                                PrepareBaseModelConfig,
                                                PrepareBaseModelParams,
                                                PrepareCallbacksConfig,
                                                TrainingParams,
                                                TrainingConfig,
                                                EvaluationConfig,
                                                EvaluationParams
                                                )


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_secrets(self) -> KaggleToken:
        # read lazily: only stage 01 needs this file, and it must not exist as a dependency elsewhere
        secrets = read_yaml(TOKEN_KEY_PATH).kaggle_secrets
        return KaggleToken(username=secrets.username, key=secrets.key)

    def get_split_params(self) -> dict:
        p = self.params
        return dict(train_ratio=p.TRAIN_RATIO, val_ratio=p.VAL_RATIO,
                    test_ratio=p.TEST_RATIO, seed=p.SEED)    
    
    def get_data_ingestion_cls_idx(self) -> ImageData:
        params = self.params
        data_ingestion_cls_idx = ImageData(
            CLASS_TO_IDX = params.CLASS_TO_IDX,
            IMG_EXTNS = params.IMG_EXTNS
        )
        return data_ingestion_cls_idx

    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
            config = self.config.prepare_base_model            
            create_directories([config.root_dir])    
            prepare_base_model_config = PrepareBaseModelConfig(
                root_dir = Path(config.root_dir),
                base_model_path = Path(config.base_model_path),
                updated_base_model_path = Path(config.updated_base_model_path)            
            )
            return prepare_base_model_config
    
    
    def get_prepare_base_model_params(self) -> PrepareBaseModelParams:
        params = self.params        
        prepare_base_model_params = PrepareBaseModelParams(
            IMG_SIZE=params.IMG_SIZE,
            BATCH_SIZE=params.BATCH_SIZE,
            INCLUDE_TOP=params.INCLUDE_TOP,
            TRAINING=params.TRAINING,
            LEARNINGRATE_P1=params.LEARNINGRATE_P1,
            LEARNINGRATE_P2=params.LEARNINGRATE_P2,
            DROPOUT=params.DROPOUT,
            ROTATION_FACTOR=params.ROTATION_FACTOR,
            BRIGHTNESS_FACTOR=params.BRIGHTNESS_FACTOR,
            CONTRAST_FACTOR=params.CONTRAST_FACTOR
            )
        return prepare_base_model_params
    

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        create_directories([
            Path(config.checkpoint_phase1_filepath).parent,
            Path(config.checkpoint_phase2_filepath).parent,
            Path(config.tensorboard_root_log_dir),
        ])
        return PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_phase1_filepath=Path(config.checkpoint_phase1_filepath),
            checkpoint_phase2_filepath=Path(config.checkpoint_phase2_filepath),
        )    
    

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


    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        create_directories([Path(training.root_dir)])
        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(self.config.prepare_base_model.updated_base_model_path),
            splits_dir=Path(training.splits_dir),
            # same yaml keys the callbacks write to -> single source of truth
            phase1_weights_path=Path(self.config.prepare_callbacks.checkpoint_phase1_filepath),
            phase2_weights_path=Path(self.config.prepare_callbacks.checkpoint_phase2_filepath),
        )
    
    def get_training_params(self) -> TrainingParams:
        p = self.params
        return TrainingParams(
            IMG_SIZE=p.IMG_SIZE, BATCH_SIZE=p.BATCH_SIZE,
            EPOCHS_P1=p.EPOCHS_P1, EPOCHS_P2=p.EPOCHS_P2,
            LEARNINGRATE_P1=p.LEARNINGRATE_P1, LEARNINGRATE_P2=p.LEARNINGRATE_P2,
            WARMUP_EPOCHS_P1=p.WARMUP_EPOCHS_P1, WARMUP_EPOCHS_P2=p.WARMUP_EPOCHS_P2,
            N_UNFREEZE=p.N_UNFREEZE, PATIENCE=p.PATIENCE,
            ROTATION_FACTOR=p.ROTATION_FACTOR, BRIGHTNESS_FACTOR=p.BRIGHTNESS_FACTOR,
            CONTRAST_FACTOR=p.CONTRAST_FACTOR,
        )  


    def get_evaluation_config(self) -> EvaluationConfig:
            c = self.config.evaluation
            create_directories([Path(c.root_dir), Path(c.report_path).parent])
            return EvaluationConfig(
                root_dir=Path(c.root_dir),
                trained_model_path=Path(c.trained_model_path),
                splits_dir=Path(c.splits_dir),
                report_path=Path(c.report_path),
                metadata_path=Path(c.metadata_path),
            )
    
    
    def get_evaluation_params(self) -> EvaluationParams:
        p = self.params
        return EvaluationParams(
            IMG_SIZE=p.IMG_SIZE, BATCH_SIZE=p.BATCH_SIZE,
            CLASS_TO_IDX=dict(p.CLASS_TO_IDX), THRESHOLD_BETA=p.THRESHOLD_BETA,
        )