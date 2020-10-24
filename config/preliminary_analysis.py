output_directory = ""

datasets = [
	{
		"name": "Wikipedia personal attacks",
		"prefix": "personal_attacks",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		'classes': [0, 1]
	},
	# ###########################################################################
	{
		"name": "Stanford treebank",
		"prefix": "stanford_treebank",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		"default_py": 0.5,
		'classes': ['neg', 'pos']
	},
	#
	# #######################################################################
	{
		"name": "Movie reviews",
		"prefix": "movies",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		'classes': ['NEG', 'POS']
	},

	{
		"name": "MultiRC",
		"prefix": "multirc",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		'classes': [False, True]
	},
	{
		"name": "FEVER",
		"prefix": "fever",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		'classes': ['REFUTES', 'SUPPORTS']
	},
	{
		"name": "E-SNLI",
		"prefix": "esnli",
		"train_path": "",
		"dev_path": "",
		"test_path": "",
		'classes': ['contradiction', 'entailment', 'neutral']
	},
]
