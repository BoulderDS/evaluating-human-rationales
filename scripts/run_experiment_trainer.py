import logging

from choose_gpu import choose_and_set_available_gpus
choose_and_set_available_gpus()
import sys
# sys.exit(0)
from transformers import Trainer, TrainingArguments, EvalPrediction, PretrainedConfig
from typing import Callable, Dict
import sklearn.metrics as mt
import numpy as np

from emnlp20.config import data_config as config
from emnlp20.dataset.dataset import prepare_data
from emnlp20.config.trainer_config import training_args_config, tunable_training_args
from emnlp20.config.model_config import model_dict, model_info
from emnlp20.util.param_combo import get_param_combos
from emnlp20.train_eval.feature_caching import get_and_save_features
from emnlp20.dataset.dataset import create_test_dataloader, create_dataloader
from emnlp20.train_eval.create_fidelity_curves import create_fidelity_curves
from emnlp20.train_eval.eval_pytorch import eval_fn
from emnlp20.util.saving_utils import copy_features

import os
import torch
import json

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.DEBUG,
)

import warnings

warnings.filterwarnings("ignore")


def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
	def compute_metrics_fn(p: EvalPrediction):
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		# if output_mode == "classification":
		# 	preds = np.argmax(preds, axis=1)
		# else:  # regression
		# 	preds = np.squeeze(preds)
		preds = np.argmax(preds, axis=1)
		return {"acc": mt.accuracy_score(p.label_ids, preds)}

	return compute_metrics_fn


OUTPUT_DIR = config.OUTPUT_DIR


DATASET_DICT = config.dataset_dict
DATASET_INFO = config.dataset_info
# TRAINING_PARAM_DICT = config.training_param_dict

CACHING_FLAG = config.CACHING_FLAG
EPOCH_LEVEL_CACHING = config.EPOCH_LEVEL_CACHING
TRAIN_FLAG = config.TRAIN_FLAG

CREATE_FIDELITY_CURVES = config.CREATE_FIDELITY_CURVES
NUM_FIDELITY_CURVE_SAMPLES = config.NUM_FIDELITY_CURVE_SAMPLES
FIDELITY_OCCLUSION_RATES = config.FIDELITY_OCCLUSION_RATES

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
			prediction_caching_info = {"best_dev_acc": 0.0,
									   "path": [os.path.join(OUTPUT_DIR, os.path.join(model_name, dataset))]}
			dataset_prediction_caching_info[dataset] = prediction_caching_info

		for param_combo in param_combos:
			dataset = DATASET_INFO[param_combo["params"][0]["dataset"]]
			model_dict = model_info[model_name]
			tunable_model_args = param_combo["params"][1]
			tunable_training_args = param_combo["params"][2]
			output_dir = os.path.join(OUTPUT_DIR, os.path.join(model_name, param_combo["name"]))
			best_model_save_path = os.path.join(OUTPUT_DIR,
												os.path.join(model_name, param_combo["params"][0]["dataset"]))
			# Model Class
			model_config = PretrainedConfig(
				max_length=dataset["max_len"],
				num_labels=len(dataset["classes"]),
				**tunable_model_args)

			candidate_model = model_dict["class"](config=model_config)

			# Get the data and create Dataset objects
			if TRAIN_FLAG:
				train_dataset, eval_dataset = prepare_data(
					model=candidate_model,
					return_dataset=True,
					**dataset)

				training_args_config["per_device_train_batch_size"] = dataset["batch_size"]
				# Save every epoch checkpoint which could be used for analysis later
				save_steps = len(train_dataset) // training_args_config['per_device_train_batch_size']

				training_args = TrainingArguments(
					output_dir=output_dir,
					save_steps=save_steps,
					**training_args_config,
					**tunable_training_args)

				trainer = Trainer(
					model=candidate_model,
					args=training_args,
					train_dataset=train_dataset,
					eval_dataset=eval_dataset,
					compute_metrics=build_compute_metrics_fn(),
					# tokenizer=candidate_model.tokenizer
				)

				# Training, Evaluating and Saving
				if config.TRAIN_FLAG:
					print(
						f"===============Training on Dataset: {dataset['name']} and param combo: {param_combo['name']}===================")
					trainer.train()
					trainer.save_model(output_dir=best_model_save_path)
					# Evaluate all epochs and save the best one


			# Caching features for analysis
			if CACHING_FLAG:
				LOAD_DIR = os.path.join(OUTPUT_DIR, os.path.join(model_name, param_combo["name"]))
				# LOAD_DIR_LIST = [LOAD_DIR]
				LOAD_DIR_LIST = []
				# LOAD_DIR_LIST.append(best_model_save_path)
				test_dataloader = create_test_dataloader(
					model=candidate_model,
					filepath=dataset["test_path"],
					classes=dataset["classes"],
					batch_size=training_args_config["per_device_eval_batch_size"]
				)

				if CREATE_FIDELITY_CURVES:
					print(f'Creating fidelity curves with {NUM_FIDELITY_CURVE_SAMPLES} sample(s) each for occlusion rates: \n{FIDELITY_OCCLUSION_RATES}')

					model_load_path = os.path.join(LOAD_DIR, 'pytorch_model.bin')
					cache_model = model_dict["class"](config=model_config)
					cache_model.load_state_dict(torch.load(model_load_path))

					create_fidelity_curves(
						model=cache_model,
						dataset_path=dataset["test_path"],
						dataset_classes=dataset["classes"],
						batch_size=training_args_config["per_device_eval_batch_size"],
						output_dir=os.path.join(LOAD_DIR, 'fidelity_curves'),
						num_samples=NUM_FIDELITY_CURVE_SAMPLES,
						occlusion_rates=FIDELITY_OCCLUSION_RATES
					)

				if EPOCH_LEVEL_CACHING:
					LOAD_DIR_LIST = LOAD_DIR_LIST + [
						os.path.join(LOAD_DIR, name) for name in os.listdir(LOAD_DIR) if "checkpoint-" in name
					]

					# Save features for epoch zero
					cache_model = model_dict["class"](config=model_config)
					get_and_save_features(
						test_dataloader=test_dataloader,
						model=cache_model,
						tokenizer=cache_model.tokenizer,
						save_dir=os.path.join(LOAD_DIR, "epoch-0"),
					)

				for load_path in LOAD_DIR_LIST:
					print(
						f"===============Feature caching on Dataset: {dataset['name']} and"
						f" param combo: {param_combo['name']}, load path {load_path} ==================="
					)

					# cache_model = RobertaClassifier.from_pretrained(load_path)
					model_load_path = os.path.join(load_path, 'pytorch_model.bin')
					with open(os.path.join(load_path, 'config.json'), 'r')as f:
						saved_config = json.load(f)
					saved_config = PretrainedConfig(
						num_labels=len(dataset["classes"]),
						**saved_config
					)
					cache_model = model_dict["class"](config=saved_config)
					cache_model.load_state_dict(torch.load(model_load_path))

					# look at the output _dir
					get_and_save_features(
						test_dataloader=test_dataloader,
						model=cache_model,
						tokenizer=cache_model.tokenizer,
						save_dir=load_path,
					)

					# Get the epoch with best dev acc
					dev_dataloader = create_dataloader(cache_model, dataset["classes"], dataset["dev_path"], dataset["batch_size"])
					dev_acc, _ = eval_fn(cache_model, dev_dataloader, 5)

					if dev_acc > dataset_prediction_caching_info[
						param_combo["params"][0]["dataset"]]["best_dev_acc"]:
						print(f"Dataset: {param_combo['params'][0]['dataset']}| Eval Acc: {dev_acc}")
						print(f"Path: {model_load_path}")
						copy_features(load_dir=load_path, output_dir=best_model_save_path)
						dataset_prediction_caching_info[
							param_combo["params"][0]["dataset"]]["best_dev_acc"] = dev_acc
	print("Done!")


