import os
import warnings

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from choose_gpu import choose_and_set_available_gpus
from emnlp20 import experiment_config as config
from emnlp20.dataset.dataset import prepare_data
from emnlp20.train_eval.eval_pytorch import eval_fn
from emnlp20.train_eval.feature_caching import get_and_save_features
from emnlp20.train_eval.train_pytorch import train_fn
from emnlp20.util.param_combo import get_param_combos

warnings.filterwarnings("ignore")

chosen_gpu = choose_and_set_available_gpus()
device = torch.device("cuda:" + chosen_gpu)

OUTPUT_DIR = config.OUTPUT_DIR
MAX_LEN = config.MAX_LEN_FLAG
MAX_ROWS = config.MAX_SAMPLES_FLAG

MODEL_DICT = config.model_dict
MODEL_INFO = config.model_info
DATASET_DICT = config.dataset_dict
DATASET_INFO = config.dataset_info
TRAINING_PARAM_DICT = config.training_param_dict

TRAIN_MODEL_FLAG = config.TRAIN_MODEL_FLAG
EVAL_MODEL_FLAG = config.EVAL_MODEL_FLAG
CACHING_FLAG = config.CACHING_FLAG

"""
for all model types:
	for all param_combos (dataset, model parameteres, training parameters)
		train
		eval
		eval_fidelity
"""

if __name__ == "__main__":
	for model_name in MODEL_DICT['model']:
		print(f"===============Training on Model: {model_name}===================")
		param_combos = get_param_combos([DATASET_DICT, MODEL_INFO[model_name]['model_param_dict'], TRAINING_PARAM_DICT])
		for param_combo in param_combos:
			dataset = DATASET_INFO[param_combo["params"][0]["dataset"]]
			model_dict = MODEL_INFO[model_name]
			candidate_model = model_dict["class"](max_len=MAX_LEN, num_labels=len(dataset["classes"]), **model_dict["model_param_dict"])
			candidate_model.to(device)

			print(f"===============Training on Dataset: {dataset['name']}===================")
			train_dataloader, dev_dataloader, test_dataloader = prepare_data(model=candidate_model,
																			 data_dir=dataset["dataset_dir"],
																			 classes=dataset['classes'],
																			 files=[dataset['train_path'],
																					dataset['dev_path'],
																					dataset['test_path']],
																			 max_rows=MAX_ROWS,
																			 batch_size=dataset["batch_size"],
																			 max_len=MAX_LEN)
			num_training_steps = int(len(train_dataloader) * param_combo['params'][2]['num_epochs'])
			optimizer = AdamW(candidate_model.parameters(), lr=param_combo['params'][2]['learning_rate'],
							  eps=1e-8)
			scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50,
														num_training_steps=num_training_steps)

			if TRAIN_MODEL_FLAG:
				SAVE_DIR = os.path.join(OUTPUT_DIR, os.path.join(model_name, param_combo['name']))
				train_fn(train_dataloader, dev_dataloader, candidate_model, tokenizer=candidate_model.tokenizer,
						 optimizer=optimizer, scheduler=scheduler,
						 n_epochs=param_combo['params'][2]['num_epochs'], save_dir=SAVE_DIR, device=device)

			if EVAL_MODEL_FLAG:
				LOAD_DIR = os.path.join(OUTPUT_DIR, os.path.join(model_name, param_combo['name']))
				try:
					if not TRAIN_MODEL_FLAG:
						candidate_model.load_model(LOAD_DIR)
					test_evals = eval_fn(candidate_model, test_dataloader,
										 nth_epoch=param_combo['params'][2]['num_epochs'])
					print(f'Model: {model_name} | Dataset: {dataset["name"]} | Test Accuracy {test_evals[0]}')
				except Exception as e:
					print("Exception occured: " + str(e))

			if CACHING_FLAG:
				LOAD_DIR = os.path.join(OUTPUT_DIR, os.path.join(model_name, param_combo['name']))
				if not TRAIN_MODEL_FLAG and not EVAL_MODEL_FLAG:
					candidate_model.load_model(LOAD_DIR)
				FEATURE_CACHE_PATH = os.path.join(OUTPUT_DIR, os.path.join(model_name, param_combo['name']))
				get_and_save_features(test_dataloader, candidate_model, candidate_model.tokenizer,
									  save_dir=FEATURE_CACHE_PATH)

	print("Done!")
