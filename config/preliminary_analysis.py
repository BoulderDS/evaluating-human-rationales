output_directory = "/data/anirudh/output/evaluating_human_rationales/"

datasets = [

	# {
	# 	"name":"Synthetic neutral/bad dataset",
	# 	"prefix":"synthetic_neutral_bad",
	# 	"train_path": "/data/sam/synthetic/train.csv",
	# 	"dev_path": "/data/sam/synthetic/dev.csv",
	# 	# "dev_rationale_path": "/data/sam/jigsaw_toxicity/personal_attacks/wiki_attack_dev_rationale_2_3_mv.csv",
	# 	"test_path": "/data/sam/synthetic/test.csv",
	# 	# "test_rationale_path": "/data/sam/jigsaw_toxicity/personal_attacks/wiki_attack_test_rationale_2_3_mv.csv",
	# 	# "output_dir": "/data/sam/output/personal_attacks",
	# 	'classes':['neutral','bad'],
	# },

	{
		"name": "Wikipedia personal attacks",
		"prefix": "personal_attacks",
		"train_path": "/data/sam/jigsaw_toxicity/personal_attacks/wiki_attack_train.csv",
		"dev_path": "/data/sam/jigsaw_toxicity/personal_attacks/wiki_attack_dev.csv",
		# "dev_rationale_path": "/data/sam/jigsaw_toxicity/personal_attacks/wiki_attack_dev_rationale_2_3_mv.csv",
		"test_path": "/data/sam/jigsaw_toxicity/personal_attacks/wiki_attack_test.csv",
		# "test_rationale_path": "/data/sam/jigsaw_toxicity/personal_attacks/wiki_attack_test_rationale_2_3_mv.csv",
		# "output_dir": "/data/sam/output/personal_attacks",
		# "default_py": 0.05,
		'classes': [0, 1]
	},
	# ###########################################################################
	{
		"name": "Stanford treebank",
		"prefix": "stanford_treebank",
		"train_path": "/data/sam/stanford_treebank/sst_train.csv",
		"dev_path": "/data/sam/stanford_treebank/sst_dev.csv",
		# "dev_rationale_path": "/data/sam/stanford_treebank/sst_dev_rationale.csv",
		"test_path": "/data/sam/stanford_treebank/sst_test.csv",
		# "test_rationale_path": "/data/sam/stanford_treebank/sst_test_rationale.csv",
		# "output_dir": "/data/sam/output/stanford_treebank",
		"default_py": 0.5,
		'classes': ['neg', 'pos']
	},
	#
	# #######################################################################
	{
		"name": "Movie reviews",
		"prefix": "movies",
		"train_path": "/data/sam/eraser/movies/train.csv",
		"dev_path": "/data/sam/eraser/movies/dev.csv",
		"test_path": "/data/sam/eraser/movies/test.csv",
		# "default_py":0.05,
		'classes': ['NEG', 'POS']
	},

	{
		"name": "MultiRC",
		"prefix": "multirc",
		"train_path": "/data/sam/eraser/multirc/train.csv",
		"dev_path": "/data/sam/eraser/multirc/dev.csv",
		"test_path": "/data/sam/eraser/multirc/test.csv",
		# "default_py":0.05,
		'classes': [False, True]
	},
	# {
	# 	"name":"CoS-E",
	# 	"prefix":"cose",
	# 	"train_path": "/data/sam/eraser/cose/train.csv",
	# 	"dev_path": "/data/sam/eraser/cose/dev.csv",
	# 	"test_path": "/data/sam/eraser/cose/test.csv",
	# 	# "default_py":0.05,
	# 	'classes':['A', 'B', 'C', 'D', 'E']
	# },
	{
		"name": "FEVER",
		"prefix": "fever",
		"train_path": "/data/sam/eraser/fever/train.csv",
		"dev_path": "/data/sam/eraser/fever/dev.csv",
		"test_path": "/data/sam/eraser/fever/test.csv",
		# "default_py":0.05,
		'classes': ['REFUTES', 'SUPPORTS']
	},
	{
		"name": "E-SNLI",
		"prefix": "esnli",
		"train_path": "/data/sam/eraser/esnli/train.csv",
		"dev_path": "/data/sam/eraser/esnli/dev.csv",
		"test_path": "/data/sam/eraser/esnli/test.csv",
		# "default_py":0.05,
		'classes': ['contradiction', 'entailment', 'neutral']
	},
	#
	# ################################################################
	#
	# {
	# 	"name":"Evidence inference",
	# 	"prefix":"evidence_inference",
	# 	"train_path": "/data/sam/eraser/evidence_inference/train.csv",
	# 	"dev_path": "/data/sam/eraser/evidence_inference/dev.csv",
	# 	"test_path": "/data/sam/eraser/evidence_inference/test.csv",
	# 	# "default_py":0.05,
	# 	'classes':['no significant difference', 'significantly decreased', 'significantly increased']
	# },
	#
	# {
	# 	"name":"BoolQ",
	# 	"prefix":"boolq",
	# 	"train_path": "/data/sam/eraser/boolq/train.csv",
	# 	"dev_path": "/data/sam/eraser/boolq/dev.csv",
	# 	"test_path": "/data/sam/eraser/boolq/test.csv",
	# 	# "default_py":0.05,
	# 	'classes':[False, True],
	# },
]

models = [
	# {
	# 	'class': LSTMModel,
	# 	'kwargs': {}
	# },

	# {
	# 	'class': RationaleModel,
	# 	'kwargs': {
	#
	#
	# 	},
	# 	'kwarg_sets':{
	# 		'hidden_size': 200,
	# 		'learning_rate': .001,
	# 		'cohesiveness_loss_weight_multiple': 2,
	# 		'sparsity_loss_weight': 0.1,
	# 		'batch_size': 512,
	# 		'dropout_rate': 0.2,
	# 		'adversarial_loss_weight': [1.0],
	# 		'attention_style':['output'],
	# 	}
	# },
	# {
	# 	'class': RationaleModel,
	# 	'kwargs': {'learning_rate': 0.001,
	# 			   'adversarial_loss_weight': 1.0,
	# 			   'cohesiveness_loss_weight_multiple': 0.0,
	# 			   'sparsity_loss_weight': 0.1,
	# 			   # 'default_py': default_py
	# 			   'name':'less_sparse_rationale_model'
	# 			   }
	# },
	# {
	# 	'class': RationaleModel,
	# 	'kwargs': {'learning_rate': 0.0005,
	# 			   'cohesiveness_loss_weight_multiple': 1.0,
	# 			   'sparsity_loss_weight': 0.15,
	# 			   'adversarial_loss_weight':0,
	# 			   'name':'no_adversary_rationale_model'},
	#
	# },
	# {
	# 	'class': AttentionModel,
	# 	'kwargs': {'mode':AttentionModel.additive}
	# },
	# {
	# 	'class': AttentionModel,
	# 	'kwargs': {'mode': AttentionModel.scaled_dot_product}
	# },

]
#
# attribution_methods = [
# 	{
# 		'name': 'attention',
# 		'function': TextClassificationModel.forward,
# 		'key': 'attention',
# 		'type': 'built-in',
# 		'test': lambda m: m.rationale,
# 	},
# {
# 	'name': 'Official LIME implementation',
# 	'function': TextClassificationModel.lime_official,
# 	'key': 'lime_official',
# 	'type': 'posthoc',
# 	'extra_kwargs': {'replacement_token': '<UNK>'}
# },

# {
# 	'name':'LIME',
# 	'function': TextClassificationModel.lime,
# 	'key':'lime',
# 	'type':'posthoc',
# 	'extra_kwargs': { 'replacement_token': '<UNK>'}
# },

# {
# 	'name': 'Simple gradients',
# 	'function': TextClassificationModel.simple_gradients,
# 	'key': 'simple_gradients',
# 	'type': 'posthoc'
# },

# {
# 	'name': 'Integrated gradients',
# 	'function': TextClassificationModel.integrated_gradients,
# 	'key': 'integrated_gradients',
# 	'type': 'posthoc'
# },
# {
# 	'name': 'Leave-one-out',
# 	'function': TextClassificationModel.leave_one_out,
# 	'key': 'leave_one_out',
# 	'type': 'posthoc',
# 	'extra_kwargs': {'replacement_token': '<UNK>'}
# }
# ]
