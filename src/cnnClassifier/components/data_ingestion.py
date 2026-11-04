import os
import zipfile
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig, KaggleToken, ImageData
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split



class DataIngestion:
    def __init__(self, config: DataIngestionConfig, token: KaggleToken, img_class_extns: ImageData):
        self.config = config
        self.secrets = token
        self.img_class_extns = img_class_extns
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            os.environ['KAGGLE_USERNAME'] = self.secrets.username
            os.environ['KAGGLE_KEY'] = self.secrets.key
            import kaggle 
            url = self.config.source_URL            
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(url,
                                             path=self.config.unzip_dir,
                                             unzip=False)
            
            logger.info(f" Casting dataset download is executed!")
        else:
            logger.info(f"File already exist and has a size of: {get_size(Path(self.config.local_data_file))}")


    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            logger.info(f"Downloaded zipfile is extracted")


    def collect_filepaths(self) -> pd.DataFrame:
        """Scan class subfolders and return a DataFrame with filepath, class name and label."""
        records = []
        for class_name, label in self.img_class_extns.CLASS_TO_IDX.items():
            class_dir = Path(os.path.join(self.config.dataset_dir, class_name))
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class folder not found: {class_dir}")
            for fp in sorted(class_dir.iterdir()):  # sorted -> reproducible order
                if fp.suffix.lower() in self.img_class_extns.IMG_EXTNS:
                    records.append({"filepath": str(fp), "class_name": class_name, "label": label})
        logger.info("data_set has been read into a pandas dataframe")
        return pd.DataFrame(records)

    @staticmethod
    def stratified_train_val_test_split(                                        
                                        df: pd.DataFrame,
                                        train_ratio: float = 0.8,
                                        val_ratio: float = 0.1,
                                        test_ratio: float = 0.1,
                                        label_col: str = "label",
                                        seed: int = 42,
                                        ):

        """Stratified split into train/val/test using two successive splits."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
            raise ValueError("Ratios must sum to 1.0")

        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=df[label_col],
            random_state=seed,
        )
        logger.info("train_set is created")

        # share of the temp set that becomes the test set
        test_share = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_share,
            stratify=temp_df[label_col],
            random_state=seed,
        )
        logger.info("test_set val_set are created")
        return (
                train_df.reset_index(drop=True),
                val_df.reset_index(drop=True),
                test_df.reset_index(drop=True),
               )

    def save_split_manifests(self, train_df, val_df, test_df) -> None:
        out_dir = self.config.manifest_path
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
            part.to_csv(out_dir / f"{name}.csv", index=False)
        logger.info(f"Manifests saved to {out_dir.resolve()}")