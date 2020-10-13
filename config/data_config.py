TRAIN_FLAG = False
CACHING_FLAG = True
EPOCH_LEVEL_CACHING = True

CREATE_FIDELITY_CURVES = False
NUM_FIDELITY_CURVE_SAMPLES = 1

FIDELITY_OCCLUSION_RATES =  [x/20 for x in range(0,21)]



# OUTPUT_DIR = "/data/anirudh/output/evaluating_human_rationales/"
OUTPUT_DIR = "/data/anirudh/output/evaluating_human_rationales/"

dataset_dict = {'dataset': ['wikiattack', 'sst', 'movies', 'multirc', 'fever', 'esnli']}
# dataset_dict = {'dataset': ['multircred', 'feverred', 'esnlired']} # movies, wikiattack
# dataset_dict = {'dataset': ["wikiattack", "sst", "multirc", "fever"]}
# dataset_dict = {'dataset': ["wikiattack"]}
# dataset_dict = {'dataset': ["movies"]}
# dataset_dict = {'dataset': ['fever']}
# dataset_dict = {'dataset': ['esnli']}
# dataset_dict = {'dataset': ['esnlired']}
# dataset_dict = {'dataset': ['wikiattackred']}


dataset_info = {
	'sst': {
		"name": "Stanford treebank",
		"data_dir": "/data/sam/stanford_treebank",
		"train_path": "/data/sam/stanford_treebank/sst_train.csv",
		"dev_path": "/data/sam/stanford_treebank/sst_dev.csv",
		"test_path": "/data/sam/stanford_treebank/sst_test.csv",
		"classes": ['neg', 'pos'],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'sstred': {
		"name": "Stanford treebank Reduced",
		"data_dir": "/data/anirudh/rationale_reduced_datasets/sst",
		"train_path": "/data/anirudh/rationale_reduced_datasets/sst/train_path.csv",
		"dev_path": "/data/anirudh/rationale_reduced_datasets/sst/dev_path.csv",
		"test_path": "/data/anirudh/rationale_reduced_datasets/sst/test_path.csv",
		"classes": ['neg', 'pos'],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'movies': {
		"name": "movie reviews",
		"data_dir": "/data/sam/eraser/movies",
		"train_path": "/data/sam/eraser/movies/train.csv",
		"dev_path": "/data/sam/eraser/movies/dev.csv",
		"test_path": "/data/sam/eraser/movies/test.csv",
		'classes': ['NEG', 'POS'],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'moviesred': {
		"name": "movie reviews Reduced",
		"data_dir": "/data/anirudh/rationale_reduced_datasets/movies",
		"train_path": "/data/anirudh/rationale_reduced_datasets/movies/train_path.csv",
		"dev_path": "/data/anirudh/rationale_reduced_datasets/movies/dev_path.csv",
		"test_path": "/data/anirudh/rationale_reduced_datasets/movies/test_path.csv",
		'classes': ['NEG', 'POS'],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'multirc': {
		"name": "MultiRC",
		"data_dir": "/data/sam/eraser/multirc",
		"train_path": "/data/sam/eraser/multirc/train.csv",
		"dev_path": "/data/sam/eraser/multirc/dev.csv",
		"test_path": "/data/sam/eraser/multirc/test.csv",
		'classes': [False, True],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'multircred': {
		"name": "MultiRC Reduced",
		"data_dir": "/data/anirudh/rationale_reduced_datasets/multirc",
		"train_path": "/data/anirudh/rationale_reduced_datasets/multirc/train_path.csv",
		"dev_path": "/data/anirudh/rationale_reduced_datasets/multirc/dev_path.csv",
		"test_path": "/data/anirudh/rationale_reduced_datasets/multirc/test_path.csv",
		'classes': [False, True],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'fever': {
		"name": "FEVER",
		"data_dir": "/data/sam/eraser/fever",
		"train_path": "/data/sam/eraser/fever/train.csv",
		"dev_path": "/data/sam/eraser/fever/dev.csv",
		"test_path": "/data/sam/eraser/fever/test.csv",
		'classes': ['REFUTES', 'SUPPORTS'],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'feverred': {
		"name": "FEVER Reduced",
		"data_dir": "/data/anirudh/rationale_reduced_datasets/fever",
		"train_path": "/data/anirudh/rationale_reduced_datasets/fever/train_path.csv",
		"dev_path": "/data/anirudh/rationale_reduced_datasets/fever/dev_path.csv",
		"test_path": "/data/anirudh/rationale_reduced_datasets/fever/test_path.csv",
		'classes': ['REFUTES', 'SUPPORTS'],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'wikiattack': {
		"name": "Wikipedia personal attacks",
		"data_dir": "/data/sam/jigsaw_toxicity/personal_attacks",
		"train_path": "/data/sam/jigsaw_toxicity/personal_attacks/wiki_attack_train.csv",
		"dev_path": "/data/sam/jigsaw_toxicity/personal_attacks/wiki_attack_dev.csv",
		"test_path": "/data/sam/jigsaw_toxicity/personal_attacks/wiki_attack_test.csv",
		'classes': [0, 1],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'wikismall': {
		"name": "Wikipedia personal attacks Small",
		"data_dir": "/data/anirudh/wikismall",
		"train_path": "/data/anirudh/wikismall/train.csv",
		"dev_path": "/data/anirudh/wikismall/dev.csv",
		"test_path": "/data/anirudh/wikismall/test.csv",
		'classes': [0, 1],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'wikismallred': {
		"name": "Wikipedia personal attacks Small Reduced",
		"data_dir": "/data/anirudh/rationale_reduced_datasets/wikismall",
		"train_path": "/data/anirudh/rationale_reduced_datasets/wikismall/train_path.csv",
		"dev_path": "/data/anirudh/rationale_reduced_datasets/wikismall/dev_path.csv",
		"test_path": "/data/anirudh/rationale_reduced_datasets/wikismall/test_path.csv",
		'classes': [0, 1],
		"batch_size": 16,
		"max_rows": None,
		"max_len": 512,
	},
	'esnli': {
		"name": "E-SNLI",
		"data_dir": "/data/sam/eraser/esnli",
		"train_path": "/data/sam/eraser/esnli/train.csv",
		"dev_path": "/data/sam/eraser/esnli/dev.csv",
		"test_path": "/data/sam/eraser/esnli/test.csv",
		'classes': ['contradiction', 'entailment', 'neutral'],
		"batch_size": 512,
		"max_rows": None,
		"max_len": 512,
	},
	'esnlired': {
		"name": "E-SNLI Reduced",
		"data_dir": "/data/anirudh/rationale_reduced_datasets/esnli",
		"train_path": "/data/anirudh/rationale_reduced_datasets/esnli/train_path.csv",
		"dev_path": "/data/anirudh/rationale_reduced_datasets/esnli/dev_path.csv",
		"test_path": "/data/anirudh/rationale_reduced_datasets/esnli/test_path.csv",
		'classes': ['contradiction', 'entailment', 'neutral'],
		"batch_size": 128,
		"max_rows": None,
		"max_len": 512,
	},
}

# models = [
# 	{
# 		'class': RobertaClassifier,
# 		'model_param_dict': {
# 			'sparsity_weight': [0.1, 0.2, 0.3],
# 			'cohesive_weight': 0.0
# 		}
# 	},
# {
# 	'class': SKLEARN
# }
# ]
