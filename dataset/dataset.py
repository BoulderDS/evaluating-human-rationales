import torch
import pandas as pd
import os
import json
import numpy as np

class Dataset(torch.utils.data.Dataset):
	def __init__(self, X, labels, attention_masks, BATCH_SIZE_FLAG=32):
		"""Initialization"""
		self.y = labels
		self.X = X
		# self.rationale = rationale
		self.attention_masks = attention_masks
		self.BATCH_SIZE_FLAG = BATCH_SIZE_FLAG

	def __len__(self):
		"""number of samples"""
		return self.X.shape[0]

	def __getitem__(self, index):
		"""Get individual item from the tensor"""
		sample = {"input_ids": self.X[index],
				  "labels": self.y[index],
				  # "rationale": self.rationale[index],
				  "attention_mask": self.attention_masks[index]
				  }
		return sample


def prepare_data(model, classes, data_dir, train_path=None, dev_path=None, test_path=None, batch_size=32, max_rows=None, max_len=512, return_dataset=False, name=None):
	"""Preparing data for training, evaluation and testing"""

	train_dataloader = create_dataloader(model, classes, train_path, max_rows=max_rows, batch_size=batch_size, max_len=max_len, return_dataset=return_dataset, name=name)
	dev_dataloader = create_dataloader(model, classes, dev_path, max_rows=max_rows, batch_size=batch_size, max_len=max_len, return_dataset=return_dataset, name=name)
	# test_dataloader = create_dataloader(model, classes, test_path, max_rows=max_rows, batch_size=batch_size,
	# max_len=max_len, return_dataset=return_dataset)
	return train_dataloader, dev_dataloader


def prepare_data_sklearn(tokenizer, train_path, dev_path, test_path, name=None, data_dir=None, classes=None,
						 batch_size=None, max_rows=None, max_len=None):
	"""Prepare data for sklearn type model"""
	train_df = create_tokenized_data(tokenizer, train_path, classes)
	eval_df = create_tokenized_data(tokenizer, dev_path, classes)
	test_df = create_tokenized_data(tokenizer, test_path, classes)
	return train_df, eval_df, test_df


def create_tokenized_data(tokenizer, filepath, classes):
	data_df = pd.read_csv(filepath)
	data_df['input_ids'], data_df['attention_mask'] = zip(*data_df['text'].map(tokenizer.tokenize))
	data_df["labels"] = data_df['classification'].apply(lambda x: classes.index(x))
	return data_df


def create_dataloader(model, classes, filepath, batch_size=32, max_rows=None, class_specific=None, max_len=512, return_dataset=False, name=None):
	"""Preparing dataloader"""
	data_df = pd.read_csv(filepath)
	data_df = data_df[data_df['text'].notna()]
	data_df.reset_index(drop=True, inplace=True)
	# convert rationale column to list from string
	try:
		data_df = data_df[data_df['rationale'].notna()]
		data_df.reset_index(drop=True, inplace=True)
		try:
			data_df["rationale"] = data_df['rationale'].apply(lambda s: json.loads(s))
		except Exception as e:
			# for handling rationale string from wikiattack
			data_df["rationale"] = data_df["rationale"].apply(lambda s: s.strip("[").strip("]").split())
	except Exception as e:
		pass
	if max_rows is not None:
		data_df = data_df.iloc[:max_rows]

	if name == "E-SNLI":
		# temp solution
		crop_len = get_crop_length(data_df)
		model.max_len = crop_len

	if name == "E-SNLI Reduced":
		# temp solution
		crop_len = get_crop_length(data_df)
		model.max_len = crop_len

	# train_df['input_ids'], train_df['attention_mask'] = train_df['text'].apply(self.tokenize)
	data_df['text']= data_df['text'].apply(lambda t:t.replace('[SEP]',model.tokenizer.sep_token))

	data_df['input_ids'], data_df['attention_mask'] = zip(*data_df['text'].map(model.tokenize))

	input_id_tensor = torch.tensor(data_df['input_ids'])
	attention_mask_tensor = torch.tensor(data_df['attention_mask'])

	# rationale_tensor = torch.tensor(data_df['rationale'].apply(lambda s: s + [0]*(model.max_len - len(s))))
	labels_tensor = create_label_tensor(data_df, classes)
	if class_specific is not None:
		# input_id_tensor, labels_tensor, rationale_tensor, attention_mask_tensor = \
		# 	reduce_data_class_specific(input_id_tensor, labels_tensor, rationale_tensor,
		# 							   attention_mask_tensor, class_specific)
		pass

	dataset_ds = Dataset(input_id_tensor, labels_tensor, attention_mask_tensor,
						 BATCH_SIZE_FLAG=batch_size)
	if return_dataset:
		return dataset_ds
	return torch.utils.data.DataLoader(dataset_ds, batch_size=dataset_ds.BATCH_SIZE_FLAG, shuffle=True)


class TestDataset(torch.utils.data.Dataset):
	def __init__(
			self, id, input_ids, attention_mask, sufficiency_input_ids, sufficiency_attention_mask,
			comprehensiveness_input_ids, comprehensiveness_attention_mask, labels, batch_size=32):
		"""Initialization"""
		self.id = id
		self.input_ids = input_ids
		self.attention_mask = attention_mask
		self.sufficiency_input_ids = sufficiency_input_ids
		self.sufficiency_attention_mask = sufficiency_attention_mask
		self.comprehensiveness_input_ids = comprehensiveness_input_ids
		self.comprehensiveness_attention_mask = comprehensiveness_attention_mask
		self.labels = labels
		self.batch_size = batch_size

	def __len__(self):
		"""number of samples"""
		return self.input_ids.shape[0]

	def __getitem__(self, index):
		"""Get individual item from the tensor"""
		sample = {
			"id": self.id[index],
			"input_ids": self.input_ids[index],
			"attention_mask": self.attention_mask[index],
			"sufficiency_input_ids": self.sufficiency_input_ids[index],
			"sufficiency_attention_mask": self.sufficiency_attention_mask[index],
			"comprehensiveness_input_ids": self.comprehensiveness_input_ids[index],
			"comprehensiveness_attention_mask": self.comprehensiveness_attention_mask[index],
			"labels": self.labels[index]
		}
		return sample


def create_test_dataloader(model,
						   filepath,
						   classes,
						   batch_size=16,
						   rationale_occlusion_rate=None,
						   ):
	"""preparing the test dataloader"""
	data_df = pd.read_csv(filepath)

	if "rationale" not in data_df.columns:
		data_df["rationale"] = data_df["text"].apply(lambda s: s.strip("[").strip("]").split())

	data_df = data_df[data_df['rationale'].notna()]
	data_df.reset_index(drop=True, inplace=True)
	try:
		data_df["rationale"] = data_df['rationale'].apply(lambda s: json.loads(s))
	except Exception as e:
		# for handling rationale string from wikiattack
		data_df["rationale"] = data_df["rationale"].apply(lambda s: s.strip("[").strip("]").split())
		data_df["rationale"] = [[float(xx) for xx in x] for x in data_df["rationale"]]

	#because SST rationale values are sometimes 0.5 and we don't want that to cause problems later
	data_df["rationale"] = data_df["rationale"].apply(binarize_rationale)

	if rationale_occlusion_rate is not None:
		print(f'Randomly occluding rationales at rate {rationale_occlusion_rate}')
		data_df['rationale'] = data_df["rationale"].apply(lambda r: occlude_rationale(r,rate=rationale_occlusion_rate))

	data_df["sufficiency_text"] = data_df[
		["text", "rationale"]].apply(lambda s: reduce_by_alpha(*s, fidelity_type="sufficiency"), axis=1)
	data_df["comprehensiveness_text"] = data_df[
		["text", "rationale"]].apply(lambda s: reduce_by_alpha(*s, fidelity_type="comprehensiveness"), axis=1)

	data_df['sufficiency_input_ids'], data_df['sufficiency_attention_mask'] =\
		zip(*data_df['sufficiency_text'].map(model.tokenize))
	data_df['comprehensiveness_input_ids'], data_df['comprehensiveness_attention_mask'] =\
		zip(*data_df['comprehensiveness_text'].map(model.tokenize))
	data_df['input_ids'], data_df['attention_mask'] = \
		zip(*data_df['text'].map(model.tokenize))

	input_id_tensor = torch.tensor(data_df['input_ids'])
	attention_mask_tensor = torch.tensor(data_df['attention_mask'])

	sufficiency_input_id_tensor = torch.tensor(data_df['sufficiency_input_ids'])
	sufficiency_attention_mask_tensor = torch.tensor(data_df['sufficiency_attention_mask'])

	comprehensiveness_input_id_tensor = torch.tensor(data_df['comprehensiveness_input_ids'])
	comprehensiveness_attention_mask_tensor = torch.tensor(data_df['comprehensiveness_attention_mask'])

	labels_tensor = create_label_tensor(data_df, classes)

	test_dataset_ds = TestDataset(
		id=data_df["id"],
		input_ids=input_id_tensor,
		attention_mask=attention_mask_tensor,
		sufficiency_input_ids=sufficiency_input_id_tensor,
		sufficiency_attention_mask=sufficiency_attention_mask_tensor,
		comprehensiveness_input_ids=comprehensiveness_input_id_tensor,
		comprehensiveness_attention_mask=comprehensiveness_attention_mask_tensor,
		labels=labels_tensor,
		batch_size=batch_size
	)

	test_dataloader = torch.utils.data.DataLoader(
		test_dataset_ds, batch_size=test_dataset_ds.batch_size, shuffle=True)

	return test_dataloader


def binarize_rationale(rationale):
	rationale = [1.0 if x >= 0.5 else 0.0 for x in rationale]
	return rationale


def occlude_rationale(rationale, rate):
	mask = (np.random.random(len(rationale)) < rate).astype(float)

	occluded_rationale = [ri*mi for ri, mi in zip(rationale, mask)]
	return occluded_rationale


def reduce_by_alpha(text, rationale, fidelity_type="sufficiency"):
	reduced_text = ""
	# whitespace tokenization
	tokens = text.split()

	for idx in range(len(tokens)):
		try:
			if fidelity_type == "sufficiency" and rationale[idx] >= 0.5:
				reduced_text = reduced_text + tokens[idx] + " "
			elif fidelity_type == "comprehensiveness" and rationale[idx] < 0.5:
				reduced_text = reduced_text + tokens[idx] + " "
		except Exception as e:
			if fidelity_type == "comprehensiveness":
				reduced_text = reduced_text + tokens[idx] + " "

	# removed the last space from the text
	if len(reduced_text) > 0:
		reduced_text = reduced_text[:-1]

	return reduced_text


def create_label_tensor(data_df, classes):
	return torch.tensor(data_df['classification'].apply(lambda x: classes.index(x)))


def reduce_data_class_specific(input_id_tensor, labels_tensor, rationale_tensor, attention_mask_tensor, class_specific):
	class_indices = [i for i, x in enumerate(labels_tensor) if x == class_specific]
	labels_tensor = torch.tensor([labels_tensor[i].item() for i in class_indices])
	input_id_tensor = torch.tensor([input_id_tensor[i].item() for i in class_indices])
	rationale_tensor = torch.tensor([rationale_tensor[i].item() for i in class_indices])
	attention_mask_tensor = torch.tensor([attention_mask_tensor[i].item() for i in class_indices])
	return input_id_tensor, labels_tensor, rationale_tensor, attention_mask_tensor


def get_crop_length(data_df):
	data_df["split_text"] = data_df["text"].str.split()
	crop_len = data_df["split_text"].apply(len).max()
	return crop_len


def create_test_data_sklearn(tokenizer, filepath, classes):
	"""preparing the test dataloader"""
	data_df = pd.read_csv(filepath)

	data_df = data_df[data_df['rationale'].notna()]
	data_df.reset_index(drop=True, inplace=True)
	try:
		data_df["rationale"] = data_df['rationale'].apply(lambda s: json.loads(s))
	except Exception as e:
		# for handling rationale string from wikiattack
		data_df["rationale"] = data_df["rationale"].apply(lambda s: s.strip("[").strip("]").split())
		data_df["rationale"] = [[float(xx) for xx in x] for x in data_df["rationale"]]

	data_df["sufficiency_text"] = data_df[
		["text", "rationale"]].apply(lambda s: reduce_by_alpha(*s, fidelity_type="sufficiency"), axis=1)
	data_df["comprehensiveness_text"] = data_df[
		["text", "rationale"]].apply(lambda s: reduce_by_alpha(*s, fidelity_type="comprehensiveness"), axis=1)
	data_df["null_diff_text"] = data_df[
		["text", "rationale"]].apply(lambda s: reduce_by_alpha(*s, fidelity_type="null_diff"), axis=1)

	data_df['sufficiency_input_ids'], data_df['sufficiency_attention_mask'] =\
		zip(*data_df['sufficiency_text'].map(tokenizer.tokenize))
	data_df['comprehensiveness_input_ids'], data_df['comprehensiveness_attention_mask'] =\
		zip(*data_df['comprehensiveness_text'].map(tokenizer.tokenize))
	data_df['null_diff_input_ids'], data_df['null_diff_attention_mask'] = \
		zip(*data_df['null_diff_text'].map(tokenizer.tokenize))

	data_df['input_ids'], data_df['attention_mask'] = \
		zip(*data_df['text'].map(tokenizer.tokenize))

	data_df["labels"] = data_df['classification'].apply(lambda x: classes.index(x))

	return data_df


def reduce_and_save_data(dataset_info, save_dir):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	reduce_df_text_by_rationale(dataset_info["train_path"], os.path.join(save_dir, "train_path.csv"))
	reduce_df_text_by_rationale(dataset_info["dev_path"], os.path.join(save_dir, "dev_path.csv"))
	reduce_df_text_by_rationale(dataset_info["test_path"], os.path.join(save_dir, "test_path.csv"))
	return


def reduce_df_text_by_rationale(filepath, save_path):
	data_df = pd.read_csv(filepath)
	try:
		data_df["rationale"] = data_df['rationale'].apply(lambda s: json.loads(s))
	except Exception as e:
		data_df["rationale"] = data_df["rationale"].apply(lambda s: s.strip("[").strip("]").split())
		data_df["rationale"] = [[float(xx) for xx in x] for x in data_df["rationale"]]
	data_df["text"] = data_df[
		["text", "rationale"]].apply(lambda s: reduce_by_alpha(*s, fidelity_type="sufficiency"), axis=1)
	data_df["rationale"] = data_df['rationale'].apply(lambda s: json.dumps(s))
	data_df.to_csv(save_path)
