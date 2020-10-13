import logging

from choose_gpu import choose_and_set_available_gpus
choose_and_set_available_gpus(max_utilization=15000)

from transformers import EvalPrediction, PretrainedConfig

from typing import Callable, Dict
import sklearn.metrics as mt
import numpy as np

# from emnlp20.config import data_config as config
from emnlp20.dataset.dataset import prepare_data
# from emnlp20.config.trainer_config import training_args_config, tunable_training_args
# from emnlp20.config.model_config import model_dict, model_info
from emnlp20.config.sam_config import config as experiment_config
from emnlp20.util.param_combo import get_param_combos
from emnlp20.train_eval.feature_caching import get_and_save_features

import os
import torch

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




"""
for all model types:
	for all param_combos (dataset, model parameteres, training parameters)
		train
		eval
		eval_fidelity
"""
# from transformers import EvalPrediction


def main():
	OUTPUT_DIR = experiment_config['experiment']['OUTPUT_DIR']

	# DATASET_DICT = config.dataset_dict
	# DATASET_INFO = config.dataset_info
	# TRAINING_PARAM_DICT = config.training_param_dict

	CACHING_FLAG = experiment_config['experiment']['CACHING_FLAG']
	EPOCH_LEVEL_CACHING = experiment_config['experiment']['EPOCH_LEVEL_CACHING']


	for dataset in experiment_config['datasets']:
		for model_trainer_config in experiment_config['models_trainers']:
		# for model_name in model_dict['model']:
			model_name = str(model_trainer_config['model']['name'])
			logger.debug(f"===============Training on Model: {model_name}===================")
			tunable_model_args = model_trainer_config['model']["tunable_args"]
			tunable_training_args = model_trainer_config['trainer']['tunable_args']
			param_combos = get_param_combos([tunable_model_args, tunable_training_args])
			for param_combo in param_combos:
				tunable_model_args, tunable_training_args = param_combo['params']
				static_training_args = model_trainer_config['trainer']['static_args']
				model_dict = model_trainer_config['model']
				output_dir = os.path.join(OUTPUT_DIR, os.path.join(model_name, param_combo["name"]))

				# Model Class
				model_config = PretrainedConfig(
					max_length=dataset["max_len"],
					num_labels=len(dataset["classes"]),
					**tunable_model_args)

				candidate_model = model_dict["class"](config=model_config)

				# Get the data and create Dataset objects
				train_dataset, eval_dataset = prepare_data(
					model=candidate_model,
					return_dataset=True,
					**dataset)

				static_training_args["per_device_train_batch_size"] = dataset["batch_size"]
				# Save every epoch checkpoint which could be used for analysis later
				save_steps = len(train_dataset) // static_training_args['per_device_train_batch_size']


				trainer = model_trainer_config['trainer']['class'](
					output_dir=output_dir,
				 	save_steps=save_steps,
					model=candidate_model,
					train_dataset=train_dataset,
					eval_dataset=eval_dataset,
					metric_fn=build_compute_metrics_fn(),
					**static_training_args,
					**tunable_training_args
				)

				# Training, Evaluating and Saving
				if trainer.do_train:
					print(
						f"===============Training on Dataset: {dataset['name']} and param combo: {param_combo['name']}===================")
					trainer.train()
					trainer.save_model()

				# Caching features for analysis
				# This isn't really "feature caching", this is making predictions
				if CACHING_FLAG:
					LOAD_DIR = os.path.join(OUTPUT_DIR, os.path.join(model_name, param_combo["name"]))
					LOAD_DIR_LIST = [LOAD_DIR]

					if EPOCH_LEVEL_CACHING:
						LOAD_DIR_LIST = LOAD_DIR_LIST + [
							os.path.join(LOAD_DIR, name) for name in os.listdir(LOAD_DIR) if "checkpoint-" in name
						]

					for load_path in LOAD_DIR_LIST:
						print(
							f"===============Feature caching on Dataset: {dataset['name']} and"
							f" param combo: {param_combo['name']}==================="
						)

						# cache_model = RobertaClassifier.from_pretrained(load_path)
						model_load_path = os.path.join(load_path, 'pytorch_model.bin')
						cache_model = model_dict["class"](config=model_config)
						cache_model.load_state_dict(torch.load(model_load_path))

						get_and_save_features(
							test_data_path=dataset["test_path"],
							classes=dataset["classes"],
							model=cache_model,
							tokenizer=cache_model.tokenizer,
							save_dir=load_path,
							batch_size=static_training_args["per_device_eval_batch_size"]
						)
					# Save features for epoch zero
					cache_model = model_dict["class"](config=model_config)
					get_and_save_features(
						test_data_path=dataset["test_path"],
						classes=dataset["classes"],
						model=cache_model,
						tokenizer=cache_model.tokenizer,
						save_dir=os.path.join(LOAD_DIR, "epoch-0"),
						batch_size=static_training_args["per_device_eval_batch_size"]
					)
	print("Done!")

if __name__ == '__main__':
	main()