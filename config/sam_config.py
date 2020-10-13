from emnlp20.model.roberta_classifier import RobertaClassifier
from emnlp20.model.sklearn_classifier import SKLearnClassifier, RandomForestSKLearnClassifier, LogisticRegressionSKLearnClassifier
from emnlp20.train_eval.train_pytorch import TransformerTrainer
from emnlp20.train_eval.train_sklearn import SKLearnTrainer

'''
I recommend that you keep the entire config in one file
'''

config = {
	'experiment': {
		'CACHING_FLAG': True,
		'EPOCH_LEVEL_CACHING': True,
		'OUTPUT_DIR': "/data/sam/mita/output/sam_sebug/"
	},
	'datasets': [
		{
			"name": "Stanford treebank",
			"data_dir": "/data/sam/stanford_treebank",
			"train_path": "/data/sam/stanford_treebank/sst_train.csv",
			"dev_path": "/data/sam/stanford_treebank/sst_dev.csv",
			"test_path": "/data/sam/stanford_treebank/sst_test.csv",
			"classes": ['neg', 'pos'],
			"batch_size": 16,
			"max_rows": 200,
			"max_len": 512,
		}
	],
	'models_trainers': [
		# {'model': {'class': RobertaClassifier,
		#			'name':'roberta',
		# 		   'static_args': {},
		# 		   "tunable_args": {
		# 			   # "hidden_dropout_prob": [0.1, 0.2, 0.3]
		# 			   "hidden_dropout_prob": [0.1]
		# 		   }
		# 		   },
		#  'trainer': {
		# 	 'class': TransformerTrainer,
		# 	 'static_args': {
		# 		 # "output_dir": "/data/anirudh/output/evaluating_human_rationales/roberta",
		# 		 "overwrite_output_dir": True,
		# 		 "do_train": True,
		# 		 "do_eval": True,
		# 		 "do_predict": True,
		# 		 "evaluation_strategy": "steps",
		# 		 # "per_device_train_batch_size": 16,
		# 		 "per_device_eval_batch_size": 16,
		# 		 # "learning_rate": 2e-5,
		# 		 "logging_steps": 50,
		# 		 # "num_train_epochs": 5,
		# 		 "warmup_steps": 50,
		# 		 "logging_dir": "/data/sam/mita/output/logging",
		# 	 },
		# 	 'tunable_args': {
		# 		 "learning_rate": [2e-5],
		# 		 "num_train_epochs": [5],
		# 	 }}
		#  },
		{
			'model': {'class': RandomForestSKLearnClassifier,
					  'name':'random_forest',
					  'static_args': {
					  },
					  'tunable_args': {}},
			'trainer': {'class': SKLearnTrainer,
						'static_args': {
							'do_train':True
						},
						'tunable_args': {}}
		},
		{
			'model': {'class': LogisticRegressionSKLearnClassifier,
					  'name':'logistic_regression',
					  'static_args': {},
					  'tunable_args': {}},
			'trainer': {'class': SKLearnTrainer,
						'static_args': {
							'do_train':True
						},
						'tunable_args': {}}
		}
	]}
