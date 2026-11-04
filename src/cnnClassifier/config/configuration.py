from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                KaggleToken,
                                                ImageData,
                                                PrepareBaseModelConfig,
                                                PrepareBaseModelParams,
                                                PrepareCallbacksConfig
                                                )
import os


class ConfigurationManager:
    def __init__(
        self,
        secrets_filepath = TOKEN_KEY_PATH,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.secrets = read_yaml(secrets_filepath)
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
            dataset_dir=Path(config.dataset_dir),
            manifest_path=Path(config.manifest_path)
        )

        return data_ingestion_config

    
    def get_data_ingestion_secrets(self) -> KaggleToken:
        secrets = self.secrets.kaggle_secrets

        data_ingestion_secrets = KaggleToken(
            username=secrets.username,
            key=secrets.key
        )

        return data_ingestion_secrets
    
    
    def get_data_ingestion_cls_idx(self) -> ImageData:
        params = self.params

        data_ingestion_cls_idx = ImageData(
            CLASS_TO_IDX = params.CLASS_TO_IDX,
            IMG_EXTNS = params.IMG_EXTNS
        )

        return data_ingestion_cls_idx
    
    ##################################################################################################
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
            config = self.config.prepare_base_model
            
            create_directories([config.root_dir])
    
            prepare_base_model_config = PrepareBaseModelConfig(
                root_dir = Path(config.root_dir),
                base_model_path = Path(config.base_model_path),
                updated_base_model_path = Path(config.updated_base_model_path),
                manifest_path = Path(config.manifest_path)
            )
            return prepare_base_model_config
    
        
    def get_prepare_base_model_params(self) -> PrepareBaseModelParams:
        params = self.params        
        prepare_base_model_params = PrepareBaseModelParams(
            IMG_SIZE=params.IMG_SIZE,
            BATCH_SIZE=params.BATCH_SIZE,
            INCLUDE_TOP=params.INCLUDE_TOP,
            EPOCHS_P1=params.EPOCHS_P1,
            EPOCHS_P2=params.EPOCHS_P2,            
            TRAINING=params.TRAINING,
            LEARNINGRATE_P1=params.LEARNINGRATE_P1,
            LEARNINGRATE_P2=params.LEARNINGRATE_P2,
            DROPOUT=params.DROPOUT,
            ROTATION_FACTOR=params.ROTATION_FACTOR,
            BRIGHTNESS_FACTOR=params.BRIGHTNESS_FACTOR,
            CONTRAST_FACTOR=params.CONTRAST_FACTOR
            )
        return prepare_base_model_params

    ##################################################################################################

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath)
        )

        return prepare_callback_config
    
    ##################################################################################################

    