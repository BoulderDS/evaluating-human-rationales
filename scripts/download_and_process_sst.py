import pytreebank
import os
from pprint import pprint, pformat
import numpy as np
import pandas as pd
# np.set_printoptions(linewidth=1000, precision=3, suppress=True, threshold=3000)
pd.set_option('display.width', 3000)
# pd.set_option('max_colwidth',2000)
pd.set_option('precision', 3)


set_names = ['train','dev',  'test']

output_dir = '../data/sst' #change this to set different output directory

html_output_sample_size = 100

seed = 2930

def main():
	print('Downloading and parsing stanford treebank into rationale dataset')

	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)

	dataset = pytreebank.load_sst()
	for set_name in set_names:
		print('{} set'.format(set_name))
		set = dataset[set_name]
		item_dicts = []

		threshold = 1.0
		for item_num, item in enumerate(set):
			if (item_num) % 100 == 0: print('Item {}...'.format(item_num))

			item_dict = {'id':item_num}
			if item.label < 2:
				item_dict['classification'] = "neg"
			elif item.label > 2:
				item_dict['classification'] = "pos"
			elif item.label == 2:
				continue #Skip neutral items

			# item_dict['classification'] = item.label

			item_dict['original_text'] = []
			item_dict['text'] = []
			item_dict['rationale'] = []


			leaves = get_leaves(item)
			for leaf in leaves:
				# label_sum = 0.0
				# label_count = 0
				# parent = leaf
				# current_label = leaf.label
				# while parent is not None:
				# 	label_sum += parent.label
				# 	label_count += 1
				# 	if abs(parent.label - current_label) >= threshold:
				# 		current_label = parent.label
				#
				#
				# 	parent = parent.parent
				# mean_label = label_sum/label_count

				item_dict['original_text'].append(leaf.text)
				item_dict['text'].append(leaf.text.lower())
				# item_dict['rationale'].append(-(leaf.label - 2.0)/2)
				# item_dict['rationale'].append(-(mean_label - 2.0)/2)
				# item_dict['rationale'].append(-(current_label - 2.0)/2)


			count_leaves_and_extreme_descendants(item)

			phrases = []
			assemble_rationale_phrases(item, phrases)
			# print('\n'.join([str(phrase) for phrase in phrases]))

			for phrase in phrases:
				phrase_rationale = [np.abs(normalize_label(phrase.label))] * phrase.num_leaves
				item_dict['rationale'].extend(phrase_rationale)
				pass
			assert(len(item_dict['rationale']) == len(item_dict['text']))

			tokenization = []
			start, end = 0,0
			for token in item_dict['original_text']:
				end = start + len(token)
				tokenization.append([start, end])
				start = end + 1

			item_dict['tokenization'] = tokenization
			item_dict['original_text'] = ' '.join(item_dict['original_text'])
			item_dict['text'] = ' '.join(item_dict['text'])
			item_dicts.append(item_dict)

		set_df = pd.DataFrame(item_dicts)
		print('Outputing {} {} items to {}'.format(set_df.shape[0], set_name, output_dir))
		set_df[['id','original_text','text', 'classification','rationale']].to_json(os.path.join(output_dir, 'sst_{}.json'.format(set_name)),orient='records')
		# set_df[['id', 'rationale']].to_csv(os.path.join(output_dir, 'sst_{}_rationale.csv'.format(set_name)))


		# negation_df = set_df[set_df['text'].apply(lambda s: 'not' in s)]
		# sample_df = negation_df.sample(n=max(html_output_sample_size, negation_df.shape[0]),random_state=seed)


		# sample_df = set_df.sample(n=html_output_sample_size,random_state=seed)
		# print('Writing sample html for this set.')
		# sample_df['true_rationalized_text'] = sample_df[['text', 'rationale']].apply(
		# 	generate_annotated_text, axis=1)
		# write_annotated_df_to_html(sample_df, os.path.join(output_dir, '{}_sample.html'.format(set_name)))
		pass

	print('Done!')


def get_leaves(tree):
	leaves = []
	if len(tree.children) > 0:
		for child in tree.children:
			leaves += get_leaves(child)
	else:
		leaves.append(tree)
	return leaves


def rprint(tree):
	print('{}: {}'.format(tree.label, tree.text if tree.text is not None else ''))
	for child in tree.children:
		rprint(child)


def explanatory_phrase(tree):
	if len(tree.children) == 0:
		return True
	else:
		# phrase_length = tree.num_leaves
		#
		# child_labels = np.array([normalize_label(child.label) for child in tree.children])
		# expected_label = sigmoid(np.sum(child_labels))
		# normalized_label = sigmoid(normalize_label(tree.label))
		#
		# phrase_unexpectedness = abs(normalized_label - expected_label)
		# phrase_propensity = phrase_unexpectedness + 1 / phrase_length

		# print('\n'.join([str(child) for child in tree.children]))

		#if this phrase is of extreme sentiment which is not explained by a descendant
		normalized_label = normalize_label(tree.label)
		normalized_max_descendant = normalize_label(tree.max_descendant)
		normalized_min_descendant = normalize_label(tree.min_descendant)

		#if label is higher than highest descendant or lower than lowest descendant
		# if (normalized_label - normalized_max_descendant) > 0.5 or (normalized_label - normalized_min_descendant) < -0.5:
		if abs(normalized_label) > abs(normalized_max_descendant) and abs(normalized_label) > abs(normalized_min_descendant):
			return True
		else:
			return False

# def num_leaves(tree):
# 	return len([child for child in tree.general_children if len(child.children) == 0])

def assemble_rationale_phrases(tree, phrases, **kwargs):
	if explanatory_phrase(tree, **kwargs):
		phrases.append(tree)
	else:
		for child in tree.children:
			assemble_rationale_phrases(child, phrases)


def count_leaves_and_extreme_descendants(tree):

	if len(tree.children) == 0: #if is leaf
		tcount = 1
		tmax = tmin = tree.label
	else:
		tcount = 0
		child_labels = [child.label for child in tree.children]
		tmax = max(child_labels)
		tmin = min(child_labels)

		for child in tree.children:
			ccount, cmax, cmin = count_leaves_and_extreme_descendants(child)
			tcount += ccount
			tmax = max(tmax, cmax)
			tmin=min(tmin, cmin)

	tree.num_leaves=tcount
	tree.max_descendant = tmax
	tree.min_descendant = tmin

	if tree.label == 4:
		_=None
	return tcount, tmax, tmin



def sigmoid(val):
	return 1 / (1 + np.exp(-val))


def normalize_label(label):
	return (label-2)/2

if __name__ == '__main__':
	main()
