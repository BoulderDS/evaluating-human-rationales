import os
from emnlp20.config.data_config import dataset_info, dataset_dict
from emnlp20.dataset.dataset import reduce_and_save_data

OUTPUT_DIR = "/data/anirudh/rationale_reduced_datasets/"

if __name__ == "__main__":
	for dataset in dataset_dict["dataset"]:
		save_dir = os.path.join(OUTPUT_DIR, dataset)
		reduce_and_save_data(dataset_info[dataset], save_dir)
