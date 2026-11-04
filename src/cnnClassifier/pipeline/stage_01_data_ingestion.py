from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger


STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion_secrets = config.get_data_ingestion_secrets()
        get_data_ingestion_cls_idx = config.get_data_ingestion_cls_idx()
        
        data_ingestion = DataIngestion(config=data_ingestion_config,
                                        token=data_ingestion_secrets,
                                        img_class_extns=get_data_ingestion_cls_idx
                                        )
        
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        dataframe = data_ingestion.collect_filepaths()
        train_df, val_df, test_df = data_ingestion.stratified_train_val_test_split(dataframe)
        data_ingestion.save_split_manifests(train_df, val_df, test_df)




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e