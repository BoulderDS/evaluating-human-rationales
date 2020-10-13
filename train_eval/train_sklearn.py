from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import pickle
import os

class SKLearnTrainer():
	def __init__(self,
				 output_dir=None,
				 save_steps=None,
				 model=None,
				 train_dataset=None,
				 eval_dataset=None,
				 metric_fn=None,
				 do_train=None,
				 **train_args):


		# convert token id tensors into strings before sending them to a scikit learn
		# tf-idf vectorizer. Indirect, but works neatly with the existing data loading
		# infrastructure

		self.output_dir = output_dir
		self.do_train=do_train

		masked_train_X = torch.where(train_dataset.attention_masks.bool(), train_dataset.X, torch.tensor(model.tokenizer.pad_token_id))
		train_documents = [' '.join(model.tokenizer.convert_ids_to_tokens(x,skip_special_tokens=True)) for x in masked_train_X]
		self.vectorizer =TfidfVectorizer(vocabulary=model.tokenizer.get_vocab())
		self.onehot_train_X = self.vectorizer.fit_transform(train_documents)
		self.train_y = train_dataset.y

		masked_eval_X = torch.where(eval_dataset.attention_masks.bool(), eval_dataset.X, torch.tensor(model.tokenizer.pad_token_id))
		eval_documents = [' '.join(model.tokenizer.convert_ids_to_tokens(x,skip_special_tokens=True)) for x in masked_eval_X]
		self.onehot_eval_X = self.vectorizer.transform(eval_documents)
		self.eval_y = eval_dataset.y

		self.model =model

		pass

	def train(self):
		self.model.model.fit(self.onehot_train_X, self.train_y)

		pass

	def save_model(self):
		model_path = os.path.join(self.output_dir, 'sklearn_model.pkl')
		with open(model_path, 'w') as f:
			pickle.dump(self.model.model, f)

		vectorizer_path = os.path.join(self.output_dir, 'sklearn_vectorizer.pkl')
		with open(vectorizer_path, 'w') as f:
			pickle.dump(self.vectorizer. f)

