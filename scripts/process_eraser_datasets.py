from util.print_util import iprint
import json
import os
import pandas as pd
import re
import numpy as np

pd.set_option('display.width', 3000)
pd.set_option('max_colwidth',100)
pd.set_option('max_columns',50)
pd.set_option('precision', 3)
pd.options.mode.chained_assignment = None


# datasets = [
# 	{'name':'E-SNLI',
# 	 'prefix':'esnli',
# 	 'input_dir':'/home/sam/sam_data/eraser/esnli'}
# ]

dataset_dirnames = [
	# "boolq",
	# "esnli",
# "cose",
	# "cose_simplified",
# "evidence_inference",
# "fever",
"movies",
# "multirc"
]

use_bert = True

eraser_dir = "/data/sam/eraser/"
# eraser_dir = "/home/ubuntu/data/eraser/"

sets = [('test','test.jsonl'),
	('train','train.jsonl'),
		('dev','val.jsonl'),
		]


docfile = 'docs.jsonl'


'''
This script processes the eraser datasets located at http://www.eraserbenchmark.com/ and converts them into unified CSVs with a standard format


'''

debug = False

# nlp = spacy.load("en_core_web_sm")

def main():
	iprint('Processing ERASER dataset(s)')


	# dirnames = os.listdir(eraser_dir)

	for dataset_dirname in dataset_dirnames:
		dataset_dir = os.path.join(eraser_dir,dataset_dirname)
		iprint('#'*40)
		iprint('Dataset: {}'.format(dataset_dirname))
		if not os.path.isdir(dataset_dir):
			continue


		
		#Look for docs

		docfilename = [filename for filename in os.listdir(dataset_dir) if filename in ['docs','docs.jsonl']][0]

		docdicts = read_jsonlines_or_dir(os.path.join(dataset_dir, docfilename))

		doc_df = pd.DataFrame(docdicts)
		doc_df.set_index('docid',inplace=True)
		# iprint('Doc columns:\n\t{}'.format('\n\t'.join(doc_df.columns)))
		iprint('Sample doc item')
		iprint(doc_df.iloc[0],1)

		for setnum, (setname, setfile) in enumerate(sets):
			iprint('{} set: '.format(setname))
			
			set_objs = read_jsonlines(os.path.join(dataset_dir, setfile))


			# iprint(pformat(set_objs[0]))
			# 
			# break

			iprint('Processing {} items'.format(len(set_objs)))
			
			interval = len(set_objs) // 10
			for obj_num,set_obj in enumerate(set_objs):

				if (obj_num % interval) == 0: iprint('Object {} ({:.1f}%)'.format(obj_num, 100*obj_num/len(set_objs)))
				if 'docids' not in set_obj: set_obj['docids'] = None

				doc_id, query_id = find_ids(set_obj['annotation_id'],set_obj['docids'],dataset_dirname)
				set_obj['docid'] = doc_id
				set_obj['query_id'] = query_id

				original_document  =  doc_df.loc[doc_id]['document']
				# nlp_doc = nlp.tokenizer(original_document)
				if dataset_dirname == 'movies':
					original_document = original_document.replace('\x12','\'') #At least one document has apostrophes replaced with some weird unicode character
				elif dataset_dirname == 'cose':
					original_document = original_document.replace('[sep]','[SEP]') #At least one document has apostrophes replaced with some weird unicode character


				text, rationale = evidences_to_rationale(doc_id, original_document, set_obj['evidences'], all_one_after_first_sep=(dataset_dirname == 'cose'))

				#Movie queries aren't meaningful
				if query_id is not None or 'query' in set_obj and dataset_dirname != 'movies':
					if query_id is not None:
						original_query = doc_df.loc[query_id]['document']
					else:
						original_query = set_obj['query']
					# nlp_query = nlp.tokenizer(query)
					query, query_rationale = evidences_to_rationale(query_id, original_query, set_obj['evidences'])

					text += ' [SEP] ' + query
					rationale = rationale + [1] + query_rationale

				set_obj['text'] = text
				set_obj['rationale'] = rationale


				if obj_num == 794:
					print(text)
					pass


			
			set_df =pd.DataFrame(set_objs)

			set_df.rename(columns={'annotation_id':'id'},inplace=True)

			set_df = set_df[['id','classification','text','rationale']]

			mean_length = set_df['rationale'].apply(len).mean()
			iprint(f'Mean # tokens: {mean_length:.3f}')

			if setnum == 0:
				iprint('Sample data row:')
				print_row(**set_df.iloc[0])

			# iprint('Dataframe:')
			# iprint(set_df,1)

			if not debug:
				setpath = os.path.join(dataset_dir,'{}.json'.format(setname))
			else:
				setpath = os.path.join(dataset_dir, '{}_debug.json'.format(setname))


			iprint('Writing JSON to {}'.format(setpath))
			set_df.to_json(setpath,orient='records')



	iprint('Done!')



# def find_ids_apply(s, dataset=None):
# 	return find_ids(s[0], s[1],dataset)
multirc_pattern = re.compile('(.+):[0-9]+:[0-9]+')
def find_ids(annotation_id, docids, dataset):
	'''
	This function looks at the docids and evidences for an annotation, decides which is the
	:param docids: a list of size 1 or 0
	:param query: a string or None
	:param evidences: dictionaries pertaining to 1 or 2 documents
	:param doc_df:
	:return:
	'''


	if dataset == 'esnli':
		docid = annotation_id+'_premise'
		query_id = annotation_id+'_hypothesis'
	elif dataset == 'movies':
		docid=annotation_id
		query_id=None
	elif dataset == 'cose':
		docid = annotation_id + '_query'
		query_id = None
	elif dataset == 'cose_simplified':
		docid=annotation_id
		query_id=None
	elif dataset == 'multirc':
		docid = multirc_pattern.match(annotation_id).groups()[0]
		query_id = None
	elif dataset in ['boolq','evidence_inference','fever']:
		assert(len(docids) == 1)
		docid = docids[0]
		query_id = None
	else:
		raise Exception('Unrecognized dataset {}'.format(dataset))


	# document = doc_df['document'][docid]
	#
	# if query_id is not None:
	# 	query = doc_df['document'][query_id]
	#
	# doc_evidences = [evidence for evidence in evidences if evidence['docid'] == docid]
	# doc_rationale = evidences_to_rationale(document, doc_evidences)
	#
	#
	# query_evidences = [evidence for evidence in evidences if evidence['docid'] == query_id]
	# query_rationale = evidences_to_rationale(query, query_evidences)

	# rdict =  {'document':document,'document_rationale':doc_rationale,'query':query,'query_rationale':query_rationale}
	# rdict = {'docid':docid, 'query_id':query_id}

	return [docid, query_id]





# def evidences_to_rationales(docid=None, nlp_doc=None, query_id=None, nlp_query=None, evidences=None):
# 	evidences = [evidence for sublist in evidences for evidence in sublist]
#
# 	doc_evidences = [evidence for evidence in evidences if evidence['docid'] == docid]
# 	doc_rationale = evidences_to_rationale(nlp_doc, doc_evidences)
#
#
# 	query_evidences = [evidence for evidence in evidences if evidence['docid'] == query_id]
# 	query_rationale = evidences_to_rationale(nlp_query, query_evidences)
#
# 	return pd.Series({"doc_rationale":doc_rationale, "query_rationale":query_rationale})

def evidences_to_rationale(docid, doc, evidences, add_cls=False, add_sep=False, all_one_after_first_sep =False, strip_unicode=True):
	'''
	Can I assume every document consists of newline-separated sentences?
	:param document:
	:param doc_evidences:
	:return:
	'''



	tokens = doc.split() #eraser datasets are simply space-separated
	# if strip_unicode:
	# 	tokens = [unidecode(token) for token in tokens]

	doc_evidences = [evidence for sublist in evidences for evidence in sublist if evidence['docid'] == docid]

	# if len(doc_evidences) == 0:
	# 	return [doc_text, None]


	# nlp_doc = nlp.tokenizer(document)
	if docid is None:
		rationale = np.ones(len(tokens))
	else:
		rationale = np.zeros(len(tokens))

	for evidence in doc_evidences:
		rationale[evidence['start_token']:evidence['end_token']] = 1.0


	if all_one_after_first_sep:
		found_sep = False
		for i in range(len(tokens)):
			if tokens[i].lower() == '[sep]':
				found_sep = True
			if found_sep:
				rationale[i] = 1.0

	doc_text = ' '.join(tokens)

	# assert(len(doc_text.split(' ')) == len(rationale))

	return [doc_text, list(rationale)]



def read_jsonlines_or_dir(path):
	if os.path.isdir(path):
		return read_json_dir(path)
	else:
		return read_jsonlines(path)

def read_jsonlines(filepath):
	iprint('Parsing jsonl file {}'.format(filepath))
	with open(filepath,'r') as f:
		objs = [json.loads(line) for line in f.readlines()]
	iprint('{} items loaded.'.format(len(objs)),1)
	return objs


doc_fn_patterns = [
	# re.compile("(?P<class>[a-z]+)R_(?P<docid>[0-9]+)\.txt"), #movie dataset
	# re.compile("(?P<docid>[A-Z]{2}_wiki_[0-9]+_[0-9]+)"), #boolq dataset
	# re.compile("(?P<docid>.+)\.txt"), #multirc dataset
	re.compile("(?P<docid>.+)") #fever dataset and evidence_interference datasets
]
def read_json_dir(dirpath):
	iprint('Parsing doc dir {}'.format(dirpath))

	objs = []
	filenames = os.listdir(dirpath)
	for i, filename in enumerate(filenames):
		filepath = os.path.join(dirpath, filename)
		with open(filepath, 'r') as f:

			if filepath.endswith('.json'):
				obj = json.load(f)
			else:
				matched= False
				for pattern in doc_fn_patterns:
					m= re.match(pattern, filename)
					if m:
						matched = True
						obj = {'document':f.read()}
						obj.update(m.groupdict())
						objs.append(obj)
						if i == 0:
							iprint('{} --> {}'.format(filename, m.groupdict()),1)
						break

				if not matched:
					raise Exception('Script does not know how to read file {}'.format(filepath))

	iprint('{} items loaded.'.format(len(objs)), 1)
	return objs


def print_row(id=None, classification=None, text=None, rationale=None):

	
	tokens = text.split(' ')
	iprint(f'ID: {id} | class: {classification} | length: {len(tokens)}')
	iprint(' '.join([f'{token} ({zi})' for token, zi in zip(tokens, rationale)]))
	


if __name__ == '__main__':
	main()