import os

import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.width', 5000)
pd.set_option('max_colwidth', 50)
pd.set_option('max_columns', 30)
pd.set_option('precision', 3)

dataset = {
	"name": "Wikipedia personal attacks",
	"prefix": "personal_attacks",
	"train_path": "/data/sam/jigsaw_toxicity/personal_attacks/wiki_attack_train.csv",
	"dev_path": "/data/sam/jigsaw_toxicity/personal_attacks/wiki_attack_dev.csv",
	"test_path": "/data/sam/jigsaw_toxicity/personal_attacks/wiki_attack_test.csv",
	'classes': [0, 1],
	'hyperparameters': {
		'batch_size': 10
	}
}

out_dir = "/data/anirudh/wikismall"

seed = 4949


def main():
	dev_df = pd.read_csv(dataset['dev_path'], index_col=0)
	dev_df = dev_df[dev_df['rationale'].notnull()]

	test_df = pd.read_csv(dataset['test_path'], index_col=0)
	test_df = test_df[test_df['rationale'].notnull()]

	combined_df = pd.concat([dev_df, test_df], axis=0)
	print(f'Total: {combined_df.shape}')

	train_df, dev_test_df = train_test_split(combined_df, test_size=300, random_state=4949)

	dev_df, test_df = train_test_split(dev_test_df, test_size=150, random_state=4949)

	print(f'Train: {train_df.shape}; Dev: {dev_df.shape}; Test: {test_df.shape}')

	print(f'Writing output files to {out_dir}')

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	train_df.to_csv(os.path.join(out_dir, 'train.csv'))
	dev_df.to_csv(os.path.join(out_dir, 'dev.csv'))
	test_df.to_csv(os.path.join(out_dir, 'test.csv'))

	print('Done!')


if __name__ == '__main__':
	main()
