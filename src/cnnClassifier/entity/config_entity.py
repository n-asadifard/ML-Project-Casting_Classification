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

@dataclass(frozen=True)
class PrepareBaseModelParams:    
    IMG_SIZE: list
    BATCH_SIZE : int
    INCLUDE_TOP: bool
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
    checkpoint_phase1_filepath: Path
    checkpoint_phase2_filepath: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    splits_dir: Path
    phase1_weights_path: Path
    phase2_weights_path: Path


@dataclass(frozen=True)
class TrainingParams:
    IMG_SIZE: list
    BATCH_SIZE: int
    EPOCHS_P1: int
    EPOCHS_P2: int
    LEARNINGRATE_P1: float
    LEARNINGRATE_P2: float
    WARMUP_EPOCHS_P1: int
    WARMUP_EPOCHS_P2: int
    N_UNFREEZE: int
    PATIENCE: int
    ROTATION_FACTOR: float
    BRIGHTNESS_FACTOR: float
    CONTRAST_FACTOR: float

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    trained_model_path: Path
    splits_dir: Path
    report_path: Path
    metadata_path: Path

@dataclass(frozen=True)
class EvaluationParams:
    IMG_SIZE: list
    BATCH_SIZE: int
    CLASS_TO_IDX: dict
    THRESHOLD_BETA: float

