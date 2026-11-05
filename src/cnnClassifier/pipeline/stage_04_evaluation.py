from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.evaluation import Evaluation

STAGE_NAME = "Evaluation"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        # cell A: load the model, pick the threshold on VALIDATION only
        config = ConfigurationManager()
        evaluation = Evaluation(config.get_evaluation_config(), config.get_evaluation_params())
        evaluation.load_model()

        _, y_val, p_val = evaluation.predict_split("val")
        threshold, val_metrics = evaluation.select_threshold(y_val, p_val)

        # cell B: the TEST set, run once. Do not tune anything after seeing this.
        _, y_test, p_test = evaluation.predict_split("test")
        test_metrics = evaluation.evaluate_split("TEST", y_test, p_test, threshold)
        # evaluation.show_misclassified(test_df, p_test, threshold)

        # cell C: persist results and the inference metadata
        evaluation.save_report(threshold, val_metrics, test_metrics)
        evaluation.save_metadata(threshold)


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        