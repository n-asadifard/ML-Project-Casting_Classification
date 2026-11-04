from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    dataset_dir: Path
    manifest_path: Path

@dataclass(frozen=True)
class KaggleToken:
    username: str
    key: str

@dataclass
class ImageData:
    CLASS_TO_IDX: dict
    IMG_EXTNS: list

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    manifest_path: Path
    
@dataclass(frozen=True)
class PrepareBaseModelParams:    
    IMG_SIZE: list
    BATCH_SIZE : int
    INCLUDE_TOP: bool
    EPOCHS_P1: int
    EPOCHS_P2 : int    
    TRAINING : bool
    LEARNINGRATE_P1: float
    LEARNINGRATE_P2: float
    DROPOUT : float    
    ROTATION_FACTOR : float
    BRIGHTNESS_FACTOR : float
    CONTRAST_FACTOR : float


@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path