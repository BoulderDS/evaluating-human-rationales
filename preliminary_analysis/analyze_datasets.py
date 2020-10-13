from emnlp20.util.putil import iprint, iinc, idec, process_number_sequence_string, ioff, ion
import sys
import os
import pandas as pd
import numpy as np
import spacy
import json
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
import warnings

# # Relative import for all_eraser config
# train_config_path = os.getcwd() + '/../pt_modeling/train_configs/'
# sys.path.insert(0, train_config_path)
from emnlp20.config.preliminary_analysis import datasets, output_directory
# from all_eraser import datasets, output_directory

from pprint import pformat

warnings.filterwarnings('ignore')

np.set_printoptions(linewidth=1000, precision=3, suppress=True, threshold=3000)
pd.set_option('display.width', 3000)
pd.set_option('max_colwidth', 50)
pd.set_option('max_columns', 20)
pd.set_option('precision', 3)
pd.options.mode.chained_assignment = None

nlp = spacy.load("en_core_web_sm")

sample_size = 5000
seed = 9992


class DataObject:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_df, self.train_X, self.train_x_lengths, self.train_class, self.train_y_index, self.train_z, \
        self.train_z_indices, self.train_x1_lengths = read_data_csv(self.dataset["train_path"], None, None,
                                                                    self.dataset["classes"], max_rows=None)

        self.dev_df, self.dev_X, self.dev_x_lengths, self.dev_class, self.dev_y_index, self.dev_z, \
        self.dev_z_indices, self.dev_x1_lengths = read_data_csv(self.dataset["dev_path"], None, None,
                                                                self.dataset["classes"], max_rows=None)

        self.test_df, self.test_X, self.test_x_lengths, self.test_class, self.test_y_index, self.test_z, \
        self.test_z_indices, self.test_x1_lengths = read_data_csv(self.dataset["test_path"], None, None,
                                                                  self.dataset["classes"], max_rows=None)

        self.all_df = pd.concat([self.train_df, self.dev_df, self.test_df])
        self.all_df.reset_index(inplace=True)
        self.all_z_indices = self.all_df[self.all_df['rationale'].notnull()].index
        self.z_df = self.all_df.loc[self.all_z_indices]
        # self.plot_histogram()
        self.stats = self.create_stats_df()

    def mean_rationale_length(self):
        return self.z_df['rationale'].apply(lambda z: z.sum()).mean()

    def mean_rationale_percent(self):
        return self.z_df['rationale'].apply(lambda z: z.mean()).mean()

    def mean_text_length(self):
        # return self.all_df['x_lengths'].mean()
        return self.z_df['x_lengths'].mean()

    # def plot_histogram(self):
    #     rationale_percent = self.z_df['rationale'].apply(lambda z: z.mean())
    #     fig, ax = plt.subplots()
    #     ax.hist([rationale_percent], bins=20, edgecolor='black')
    #     ax.set_xlabel("Rational Percent")
    #     ax.set_ylabel("Frequency")
    #     ax.set_title(self.dataset['name'])
    #     ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=rationale_percent.size))l
    #     plt.show()

    def mean_rationale_length_class(self):
        mrlc = []  # mean_rationale length per class list
        for data_class in self.dataset['classes']:
            df = self.z_df[self.z_df.classification == data_class]
            mrlc.append(df['rationale'].apply(lambda z: z.sum()).mean())
        return mrlc

    def mean_rationale_percent_class(self):
        mrpc = []  # mean_rationale length per class list
        for data_class in self.dataset['classes']:
            df = self.z_df[self.z_df.classification == data_class]
            mrpc.append(df['rationale'].apply(lambda z: z.mean()).mean())
        return mrpc

    def mean_text_length_class(self):
        mtlc = []  # mean_rationale length per class list
        for data_class in self.dataset['classes']:
            df = self.z_df[self.z_df.classification == data_class]
            mtlc.append(df['x_lengths'].mean())
        return mtlc

    def create_stats_df(self):
        stats = {'dataset': self.dataset['name'], 'classes': self.dataset['classes'],
                 'train_rows': self.train_df.shape[0],
                 'dev_rows': self.dev_df.shape[0], 'test_rows': self.test_df.shape[0],
                 "type": "Cls" if self.dataset["name"] in ["SST", "WikiAttack", "Movie"] else "RC",
                 "mean_rationale_length": self.mean_rationale_length(), "mean_text_length": self.mean_text_length(),
                 "mean_rationale_percent": self.mean_rationale_percent(),
                 "mean_rationale_length_class": self.mean_rationale_length_class(),
                 "mean_rationale_percent_class": self.mean_rationale_percent_class(),
                 "mean_text_length_class": self.mean_text_length_class(),
                 "mean_text_length_z": self.z_df["x_lengths"].mean(),
                 "mean_text_length_all": self.all_df["x_lengths"].mean(),
                 "mean_text_length_train": self.train_df["x_lengths"].mean(),
                 "mean_text_length_dev": self.dev_df["x_lengths"].mean(),
                 "mean_text_length_test": self.test_df["x_lengths"].mean(),
                 "mean_rationale_length_z": self.z_df['rationale'].apply(lambda z: z.sum()).mean(),
                 }

        return stats


def main():
    stat_list = []
    for dataset in datasets:
        iprint(dataset['name'])
        iinc()

        dataset_obj = DataObject(dataset)
        dataset_obj.mean_rationale_length_class()

        mstats = []
        if dataset['name'] == 'Extended movie reviews':
            for i, row in test_df.iterrows():
                mstats.append({'text_length': row['x_lengths'], 'rationale_length': row['rationale'].sum()})

            with open('./movies_test_lengths.jsonl', 'w') as f:
                json.dump(mstats, f)

        iprint(dataset_obj.all_df.columns, 1)

        # z_df_sample = z_df.sample(n=min(sample_size, z_df.shape[0]), random_state=seed)

        poses = ['ADJ', 'NOUN', 'VERB']
        length_mismatches = 0
        ratio_list = []
        # 	# for index, row in z_df_sample.iterrows():
        # 	# 	pos_counts = {pos:0.01 for pos in poses+['OTHER']}
        # 	# 	rationale_pos_counts = {pos:0.01 for pos in poses+['OTHER']}
        # 	#
        # 	# 	if 'original_text' in row:
        # 	# 		nlp_doc = nlp(row['original_text'].strip())
        # 	# 	else:
        # 	# 		nlp_doc = nlp(row['text'].strip())
        # 	#
        # 	# 	if len(nlp_doc) == len(row['rationale']):
        # 	# 		for token_num, token in enumerate(nlp_doc):
        # 	# 			pos = token.pos_
        # 	# 			zi = row['rationale'][token_num]
        # 	#
        # 	# 			if pos not in poses: pos = 'OTHER'
        # 	# 			pos_counts[pos] += 1
        # 	# 			rationale_pos_counts[pos] += zi
        # 	#
        # 	#
        # 	# 		pos_counts = {pos:count/len(nlp_doc) for pos, count in pos_counts.items()}
        # 	# 		rationale_pos_counts = {pos:count/sum(row['rationale']) for pos, count in rationale_pos_counts.items()}
        # 	#
        # 	# 		ratios = {pos:rationale_pos_counts[pos]/pos_counts[pos] for pos in poses+['OTHER']}
        # 	# 		ratio_list.append(ratios)
        # 	# 		pass
        # 	# 	else:
        # 	# 		length_mismatches += 1
        #
        mean_ratios = pd.DataFrame(ratio_list).mean().to_dict()

        for pos, val in mean_ratios.items():
            stats['mean_{}_rationale_ratio'.format(pos)] = val

        iprint('Statistics for {} dataset:'.format(dataset['name']))
        iprint(pformat(dataset_obj.stats, 1))
        stat_list.append(dataset_obj.stats)
        idec()

    # 	# sets= [('train', train_df, train_z_indices),
    # 	#  ('dev', dev_df, dev_z_indices),
    # 	#  ('test', test_df, test_z_indices),
    # 	# 	('dev/test', all_df, all_z_indices)
    # 	# 	   ]
    # 	#
    # 	# for setname, df, z_indices in sets:
    # 	# 	iprint('{} set'.format(setname))
    # 	# 	iinc()
    # 	# 	iprint('{} rows'.format(df.shape[0]))
    # 	# 	iprint('Class counts:')
    # 	# 	iprint(df['classification'].value_counts(),1)
    # 	# 	iprint('Mean text length: {:.1f} tokens'.format(df['x_lengths'].mean()))
    # 	#
    # 	#
    # 	# 	z_df = df.loc[z_indices]
    # 	# 	iprint('{} rows with rationales ({:.1f}%)'.format(z_df.shape[0], 100*z_df.shape[0]/df.shape[0]))
    # 	# 	for classification in classes:
    # 	# 		iprint('Class: {}'.format(classification))
    # 	# 		iinc()
    # 	# 		class_z_df = z_df[z_df['classification'] == classification]
    # 	# 		iprint('{} items of this class in this set which have rationales'.format(class_z_df.shape[0]))
    # 	# 		mean_rationale_length = class_z_df['rationale'].apply(lambda z:z.sum()).mean()
    # 	# 		mean_rationale_perc = class_z_df['rationale'].apply(lambda z: z.mean()).mean()
    # 	# 		iprint('Mean rationale length: {:.1f} tokens.'.format(mean_rationale_length))
    # 	# 		iprint('Mean rationale fraction: {:.1f}%.'.format(100*mean_rationale_perc))
    # 	#
    # 	#
    # 	# 		idec()
    # 	# 	idec()
    # 	#
    # 	# idec()

    stats_df = pd.DataFrame(stat_list)

    # iprint('All stats:')
    # iprint(stats_df)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    stats_df.to_csv(os.path.join(output_directory, 'all_stats.csv'))

    iprint('Done!')


def read_data_csv(filepath, id_dict, max_input_length, classes, max_rows=None):
    '''
    Read and process a CSV representing one part of a dataset (e.g. the training set)
    :param filepath:
    :param id_dict:
    :param max_input_length:
    :return:
    '''
    iprint('Reading in dataset from {}'.format(filepath))
    iinc()

    if max_rows is not None:
        iprint('Only reading a max of {} rows'.format(max_rows))

    data_df = pd.read_csv(filepath, nrows=max_rows, index_col=0)
    iprint('{} dataset rows read'.format(data_df.shape[0]))
    # data_df = data_df.sample(n=50000)
    # data_df = data_df.sample(frac=1)
    # data_df.reset_index(inplace=True)
    # if 'text' not in data_df or 'tokenization' not in data_df:
    # 	data_df[['text', 'tokenization']] = data_df['original_text'].apply(process_text_to_pd)

    if not 'text' in data_df:
        iprint(
            'No "text" column, so assuming that this is a document/query-style dataset and doing preprocessing to combine them')

        # data_df['text'] = data_df[['document','query']].apply(lambda s:'[CLS] ' + s['document'] + ' [SEP] ' + s['query'] + ' [SEP]',axis=1)
        data_df['document_rationale'] = data_df['document_rationale'].apply(process_number_sequence_string)

        data_df['query_rationale'] = data_df['query_rationale'].apply(process_number_sequence_string)

        data_df[['text', 'rationale', 'rationale_indices', 'document_lengths']] = data_df[
            ['document', 'document_rationale', 'query', 'query_rationale']].apply(
            lambda s: combine_document_and_query(*s), axis=1, result_type='expand')

        pass
    else:
        iprint('"text" column found, so not doing any processing of documents/queries.')
        iprint('Processing rationales')
        if 'rationale' not in data_df:
            data_df['rationale'] = None
        else:
            data_df['rationale'] = data_df['rationale'].apply(process_number_sequence_string)

        iprint('Calculating rationale indices')
        data_df['rationale_indices'] = data_df['rationale'].apply(
            lambda v: np.arange(len(v)) if not np.all([pd.isnull(v)]) else v)

        if 'document_lengths' not in data_df:
            iprint('Adding empty document_lengths column')
            data_df['document_lengths'] = None

    if id_dict is not None:
        if max_input_length is not None:
            data_x = data_df['x'] = data_df['text'].apply(
                lambda x: map_to_ids(x.lower().split(' '), id_dict)[:max_input_length]).values
        else:
            data_x = data_df['x'] = data_df['text'].apply(lambda x: map_to_ids(x.lower().split(' '), id_dict)).values

        data_x_lengths = data_df['x_lengths'] = np.array([len(x) for x in data_x])
    else:
        data_x = None
        data_x_lengths = data_df['x_lengths'] = data_df['text'].apply(lambda s: len(s.lower().split(' ')))

    # truncate rationale if we have to truncate x
    data_df['rationale'] = data_df['rationale'].apply(lambda v: v[:max_input_length] if np.all(pd.notnull(v)) else v)

    data_class = data_df['classification']
    # one_hotter = OneHotEncoder(sparse=False, categories=[classes])
    # data_one_hot_class = one_hotter.fit_transform(data_df[['classification']])
    # data_y = data_df['target'].values
    data_y = data_df['y'] = data_df['classification'].apply(lambda c: classes.index(c))
    # data_y = data_df['y'].values
    # if rationale_filepath is not None:
    # 	data_df = add_rationales_to_df(data_df, rationale_filepath)
    # else:
    # 	data_df['rationale'] = None

    data_z = data_df['rationale'].values
    data_z_indices = data_df[data_df['rationale'].notnull()].index
    iprint('{} items out of {} in data set that have a rationale.'.format(data_z_indices.shape[0], data_df.shape[0]))

    idec()
    return data_df, data_x, data_x_lengths, data_class, data_y, data_z, data_z_indices, data_df['document_lengths']


def combine_document_and_query(document, document_rationale, query, query_rationale):
    if np.all(pd.isnull(query)):
        query = ''

    text = document + ' <SEP> ' + query
    document_tokens = document.split(' ')
    query_tokens = query.split(' ')

    no_document_rationale = np.all(pd.isnull(document_rationale))
    no_query_rationale = np.all(pd.isnull(query_rationale))

    if no_document_rationale and no_query_rationale:
        rationale = rationale_mask = None
    else:
        if no_document_rationale:
            document_rationale = np.zeros(len(document_tokens))
            document_rationale_mask = np.zeros(len(document_tokens))
        else:
            assert (len(document_rationale) == len(document_tokens))
            document_rationale_mask = np.ones(len(document_tokens))

        if no_query_rationale:
            query_rationale = np.zeros(len(query_tokens))
            query_rationale_mask = np.zeros(len(query_tokens))
        else:
            assert (len(query_rationale) == len(query_tokens))
            query_rationale_mask = np.ones(len(query_tokens))

        rationale = np.concatenate([document_rationale, [0], query_rationale])
        rationale_mask = np.concatenate([document_rationale_mask, [0], query_rationale_mask])

    rationale_indices = np.nonzero(rationale_mask)

    return [text, rationale, rationale_indices, len(document_tokens)]


if __name__ == '__main__':
    main()
