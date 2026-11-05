from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier.components.training import Training
from cnnClassifier.utils.common import plot_history
from cnnClassifier import logger
import keras



STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # cell A: setup
        config = ConfigurationManager()
        params = config.get_training_params()
        keras.utils.set_random_seed(config.params.SEED)

        training = Training(config=config.get_training_config(), params=params)
        training.get_base_model()
        training.build_datasets()
        prepare_callbacks = PrepareCallback(config=config.get_prepare_callback_config()) 

        # cell B: phase 1
        history_p1 = training.run_phase(1, prepare_callbacks.get_callbacks(phase=1, patience=params.PATIENCE))
        # plot_history(history_p1, "phase 1")

        # cell C: phase 2
        history_p2 = training.run_phase(2, prepare_callbacks.get_callbacks(phase=2, patience=params.PATIENCE))
        # plot_history(history_p2, "phase 2")
        training.save_trained_model()



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        