import logging

# choose_and_set_available_gpus()

import numpy as np
import pandas as pd
import pickle

from emnlp20.config import data_config as config
from emnlp20.config.trainer_config import tunable_training_args
from emnlp20.config.model_config import model_dict, model_info
from emnlp20.util.param_combo import get_param_combos
from emnlp20.model.sklearn_classifier import SklearnTokenizer
from emnlp20.dataset.dataset import create_test_data_sklearn, prepare_data_sklearn
from emnlp20.fidelity.utility import reduce, compute_fidelity
import os

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.DEBUG,
)

import warnings

warnings.filterwarnings("ignore")


OUTPUT_DIR = config.OUTPUT_DIR


DATASET_DICT = config.dataset_dict
DATASET_INFO = config.dataset_info
# TRAINING_PARAM_DICT = config.training_param_dict

CACHING_FLAG = config.CACHING_FLAG
EPOCH_LEVEL_CACHING = config.EPOCH_LEVEL_CACHING
TRAIN_FLAG = config.TRAIN_FLAG

"""
for all model types:
	for all param_combos (dataset, model parameteres, training parameters)
		train
		eval
		eval_fidelity
"""
# from transformers import EvalPrediction


if __name__ == "__main__":
	for model_name in model_dict['model']:
		logger.debug(f"===============Training on Model: {model_name}===================")
		tunable_model_args = model_info[model_name]["tunable_model_args"]
		param_combos = get_param_combos([DATASET_DICT, tunable_model_args, tunable_training_args])
		dataset_prediction_caching_info = {}
		for dataset in DATASET_DICT['dataset']:
			prediction_caching_info = {"best_dev_acc": 0.0, "path": [os.path.join(OUTPUT_DIR, os.path.join(model_name, dataset))]}
			dataset_prediction_caching_info[dataset] = prediction_caching_info

		for param_combo in param_combos:
			dataset = DATASET_INFO[param_combo["params"][0]["dataset"]]
			model_dict = model_info[model_name]
			trained_model_details = {}
			tunable_model_args = param_combo["params"][1]
			tunable_training_args = param_combo["params"][2]
			output_dir = os.path.join(OUTPUT_DIR, os.path.join(model_name, param_combo["name"]))
			best_model_save_path = os.path.join(OUTPUT_DIR, os.path.join(model_name, param_combo["params"][0]["dataset"]))

			# Get the data and create Dataset objects
			dataset_prediction_caching_info[param_combo["params"][0]["dataset"]]["path"].append(output_dir)
			# Model Class

			if TRAIN_FLAG:
				tokenizer = SklearnTokenizer(max_length=dataset["max_len"])
				train_df, eval_df, test_df = prepare_data_sklearn(tokenizer=tokenizer, **dataset)
				candidate_model = model_dict["class"](train_df, dataset["max_len"], **tunable_model_args)

				candidate_model.train(train_df=train_df)
				trained_model_details["model"] = candidate_model
				trained_model_details["dev_acc"] = candidate_model.eval(eval_df=eval_df)
				trained_model_details["params"] = param_combo
				candidate_model.save_model(save_path=output_dir)

				if trained_model_details["dev_acc"] > dataset_prediction_caching_info[
					param_combo["params"][0]["dataset"]]["best_dev_acc"]:
					candidate_model.save_model(save_path=best_model_save_path)
					dataset_prediction_caching_info[
						param_combo["params"][0]["dataset"]]["best_dev_acc"] = trained_model_details["dev_acc"]

		# Caching predictions for analysis
		if CACHING_FLAG:
			for dataset in DATASET_DICT['dataset']:
				dataset_info = DATASET_INFO[dataset]
				tokenizer = SklearnTokenizer(max_length=dataset_info["max_len"])
				test_df = create_test_data_sklearn(tokenizer, filepath=dataset_info["test_path"],
														   classes=dataset_info["classes"])
				for load_path in dataset_prediction_caching_info[dataset]["path"]:
					print(
						f"feature caching from the directory: {load_path}"
					)
					cache_model = pickle.load(open(os.path.join(load_path, "model.sav"), 'rb'))

					predicted_classes, prob_y_hat = cache_model.predict(input_ids=test_df["input_ids"])
					prob_y_hat = prob_y_hat[np.arange(len(prob_y_hat)), predicted_classes]

					_, prob_y_hat_alpha = cache_model.predict(input_ids=test_df["sufficiency_input_ids"])
					prob_y_hat_alpha = prob_y_hat_alpha[np.arange(len(prob_y_hat_alpha)), predicted_classes]

					_, prob_y_hat_alpha_comp = cache_model.predict(input_ids=test_df["comprehensiveness_input_ids"])
					prob_y_hat_alpha_comp = prob_y_hat_alpha_comp[np.arange(len(prob_y_hat_alpha_comp)), predicted_classes]

					_, prob_y_hat_0 = cache_model.predict(input_ids=test_df["null_diff_input_ids"])
					prob_y_hat_0_predicted_class = prob_y_hat_0[np.arange(len(prob_y_hat_0)), predicted_classes]
					null_diff = 1 - compute_fidelity(prob_y_hat=prob_y_hat,
														   prob_y_hat_alpha=prob_y_hat_0_predicted_class,
														   fidelity_type="sufficiency")

					feature_cache_df = pd.DataFrame({
						# 'id': id,
						'prob_y_hat': prob_y_hat,
						'prob_y_hat_alpha': prob_y_hat_alpha,
						'prob_y_hat_alpha_comp': prob_y_hat_alpha_comp,
						'null_diff': null_diff,
						'true_classes': test_df["labels"],
						'predicted_classes': predicted_classes,
						# 'zero_probs': zero_probs
					})
					print("===========================================================================================")
					print(load_path)
					feature_cache_df.to_csv(load_path + "/feature.csv")

	print("Done!")
