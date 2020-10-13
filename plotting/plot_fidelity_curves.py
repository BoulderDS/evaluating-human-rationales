import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as mt
import sys
sys.path.append('../../')
from emnlp20.fidelity.fidelity import Fidelity

sns.set(font_scale=3.0, rc={
	"lines.linewidth": 3,
	"lines.markersize":20,
	"ps.useafm": True,
	"axes.facecolor": 'white',
	"font.sans-serif": ["Helvetica"],
	"pdf.use14corefonts" : True,
	"text.usetex": False,
})

LINEWIDTH = 3
MARKERSIZE = 10
TICKLABELSIZE = 14
LEGENDLABELSIZE = 14
LABELSIZE = 23
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 1.0


pd.set_option('display.width', 5000)
pd.set_option('max_colwidth', 150)
pd.set_option('max_columns', 50)
# pd.set_option('display.max_rows', 50)
pd.set_option('max_rows', 100)
pd.set_option('min_rows', 100)

pd.set_option('precision', 3)

experiment_directory = '/data/sam/mita/output/evaluating_human_rationales/roberta/'

model_dirs = ['d=sst_hdp=0.1_lr=2e-05_nte=5',
			  'd=movies_hdp=0.1_lr=2e-05_nte=5',
			  'd=multirc_hdp=0.1_lr=2e-05_nte=5',
			  'd=fever_hdp=0.1_lr=2e-05_nte=5',
			  'd=esnli_hdp=0.1_lr=2e-05_nte=5',
			  'd=wikiattack_hdp=0.1_lr=2e-05_nte=5'
			  ]


def subdir_paths(dir_path):
	try:
		filenames = os.listdir(dir_path)
		paths = [(filename, os.path.join(dir_path, filename)) for filename in filenames]
		subdir_paths = [(filename, filepath) for filename, filepath in paths if os.path.isdir(filepath)]
		return sorted(subdir_paths, key=lambda t: t[0])
	except:
		return []


def parse_combo_name(combo_name, abb_dict=None):
	#     print(combo_name)
	pieces = [piece.split('_') for piece in combo_name.split('=')]
	#     print(pieces)
	valdict = {}
	for i in range(1,len(pieces)):
		#         print(pieces[i-1][-1])
		#         print('_'.join(pieces[i][:-1]))
		key = pieces[i-1][-1]
		value = '_'.join(pieces[i][:-1]) if i < len(pieces)-1 else '_'.join(pieces[i])
		if abb_dict is not None and key in abb_dict:
			key = abb_dict[key]
		valdict[key] = value

	return valdict



dfs=[]
for model_dir in model_dirs:
	combo = parse_combo_name(model_dir)
	print(combo)
	curve_directory = os.path.join(experiment_directory, model_dir, 'fidelity_curves')
	for occlusion_rate, occlusion_dir in subdir_paths(curve_directory):
		print('\t',occlusion_rate)
		for sample_num, sample_dir in subdir_paths(occlusion_dir):
			print('\t\t',sample_num)
			try:
				df = pd.read_csv(os.path.join(sample_dir,'feature.csv'),index_col=0)
				df['occlusion_rate'] = float(occlusion_rate)
				df['sample_num'] = int(sample_num)
				df['combo_name'] = model_dir
				for key, value in combo.items():
					df[key] = value
				dfs.append(df)
				acc = mt.accuracy_score(df['true_classes'],df['predicted_classes'])
				df['acc'] = acc
			except Exception as ex:
				print(ex)

all_df = pd.concat(dfs,axis=0)
all_df.shape

all_df['raw_sufficiency'] = all_df['prob_y_hat'] - all_df['prob_y_hat_alpha']
# all_df['clipped_sufficiency'] = 1-np.clip(all_df['raw_sufficiency'],0,1)
# all_df['clipped_0_sufficiency'] = 1-np.clip(all_df['null_diff'],0,1)

# all_df['normalized_sufficiency'] = (all_df['clipped_sufficiency']-all_df['clipped_0_sufficiency'])/(1-all_df['clipped_0_sufficiency'])
# with pd.option_context('mode.use_inf_as_na', True):
#     all_df['normalized_sufficiency'].fillna(0.0, inplace=True)


all_df['raw_comprehensiveness'] = all_df['prob_y_hat'] - all_df['prob_y_hat_alpha_comp']
# all_df['clipped_comprehensiveness'] = np.clip(all_df['raw_comprehensiveness'],0,1)
# all_df['clipped_1_comprehensiveness'] = np.clip(all_df['null_diff'],0,1)

# all_df['normalized_comprehensiveness'] = all_df['clipped_comprehensiveness']/all_df['clipped_1_comprehensiveness']
# with pd.option_context('mode.use_inf_as_na', True):
#     all_df['normalized_comprehensiveness'].fillna(0.0, inplace=True)


fidelity_calculator = Fidelity(experiment_id='fidelity_curve_plotting')
all_df['normalized_sufficiency']  = fidelity_calculator.compute(
	prob_y_hat=all_df['prob_y_hat'].values,
	prob_y_hat_alpha=all_df['prob_y_hat_alpha'].values,
	null_difference=all_df['null_diff'].values,
	normalization=True,
reduction=None)

all_df['normalized_comprehensiveness'] = fidelity_calculator.compute(
	prob_y_hat=all_df['prob_y_hat'].values,
	prob_y_hat_alpha=all_df['prob_y_hat_alpha_comp'].values,
	null_difference=all_df['null_diff'].values,
	fidelity_type="comprehensiveness",
	normalization=True,
	reduction=None)

all_df['clipped_sufficiency']  = fidelity_calculator.compute(
	prob_y_hat=all_df['prob_y_hat'].values,
	prob_y_hat_alpha=all_df['prob_y_hat_alpha'].values,
	null_difference=all_df['null_diff'].values,
	normalization=False,
	reduction=None)

all_df['clipped_comprehensiveness'] = fidelity_calculator.compute(
	prob_y_hat=all_df['prob_y_hat'].values,
	prob_y_hat_alpha=all_df['prob_y_hat_alpha_comp'].values,
	null_difference=all_df['null_diff'].values,
	fidelity_type="comprehensiveness",
	normalization=False,
	reduction=None)

meaned_across_points = all_df.groupby(['combo_name', 'occlusion_rate','sample_num','d'],as_index=False).mean()
meaned_across_points


meaned_across_samples = meaned_across_points.groupby(['combo_name', 'occlusion_rate','d'],as_index=False).mean()
meaned_across_samples

dataset_codes = ["sst",'movies','multirc','esnli','fever','wikiattack']
datasets = ["Stanford treebank", "movie reviews", "MultiRC", "E-SNLI", "FEVER", "Wikipedia personal attacks", "Wikipedia personal attacks (small)"]
datasetabs = ["SST", "Movie", "MultiRC", "E-SNLI", "FEVER", "WikiAttack", "WikiSmall"]
dataset_ab_dict = {}
for i in range(len(datasets)):
	dataset_ab_dict[datasets[i]] = datasetabs[i]
markers = ["^", "*", "<", ">", "o", "v",'v']
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

dataset_colors = {}
dataset_colors_idx = {}
dataset_markers = {}
acc_labels = {}

for i in range(len(dataset_codes)):
	dataset_colors[dataset_codes[i]] = colors[i]
	dataset_markers[dataset_codes[i]] = markers[i]
	acc_labels[dataset_codes[i]] = datasetabs[i]
	dataset_colors_idx[i] = colors[i]



# metric_names= ['sufficiency','comprehensiveness']
names_metrics = [
	('accuracy','acc'),
	('sufficiency','clipped_sufficiency'),
	('comprehensiveness','clipped_comprehensiveness')]

for name, metric in names_metrics:

	for dataset in meaned_across_samples['d'].unique():
		dataset_df = meaned_across_samples[meaned_across_samples['d'] == dataset]
		dataset_df['occlusion_rate'] = 1-dataset_df['occlusion_rate']
		plt.errorbar(dataset_df['occlusion_rate'],
					 dataset_df[metric],
					 #                  yerr=error,
					 color=dataset_colors[dataset],
					 marker=dataset_markers[dataset],
					 linestyle="solid",
					 ecolor=dataset_colors[dataset],
					 markersize=MARKERSIZE,
					 label=acc_labels[dataset])


	plt.legend(loc="upper left", bbox_to_anchor=(1.04,1), fontsize=LEGENDLABELSIZE)
	plt.xlabel('occlusion', fontsize=LABELSIZE)
	plt.ylabel(name, fontsize=LABELSIZE)
	plt.tick_params(axis="x", labelsize=TICKLABELSIZE)
	plt.tick_params(axis="y", labelsize=TICKLABELSIZE)
	# plt.savefig("/data/anirudh/output/train_model_debugging/figs/parsimony/sufficiency.pdf", bbox_inches = 'tight', dpi=300)
	plt.show()