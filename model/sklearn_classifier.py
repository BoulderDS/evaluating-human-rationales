from transformers import PretrainedConfig, RobertaTokenizerFast
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics as mt
import pickle
import os


class SKLearnClassifier:

	def __init__(
			self,
			model_class: type,
			train_df,
			max_length=512,
			**tunable_model_args
	):
		self.model = model_class(verbose=0, n_jobs=20, **tunable_model_args)
		self.tokenizer = SklearnTokenizer(max_length=max_length)
		self.vectoriser = TfidfVectorizer()
		self.max_length = max_length
		self.fit_vectorizer(train_df)

	def train(self, train_df=None, attention_mask=None, rationale=None):
		# use tokenizer to convert input_ids into raw texts (using convert_ids_to_tokens)
		# use (already-fit) vectorizer to convert raw texts into numpy matrix
		# run self.model.predict on numpy matrix
		raw_text = [self.tokenizer.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in train_df["input_ids"]]
		train_x = self.vectoriser.transform([" ".join([str(w) for w in ins]) for ins in raw_text])
		self.model.fit(train_x, train_df["labels"])

	def eval(self, eval_df=None):
		raw_text = [self.tokenizer.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in eval_df["input_ids"]]
		eval_x = self.vectoriser.transform([" ".join([str(w) for w in ins]) for ins in raw_text])
		eval_py = self.model.predict(eval_x)

		dev_acc = mt.accuracy_score(eval_py, eval_df["labels"])
		return dev_acc

	def predict(self, input_ids=None):
		raw_text = [self.tokenizer.tokenizer.convert_ids_to_tokens(indi_input_id) for indi_input_id in input_ids]
		test_x = self.vectoriser.transform([" ".join([str(w) for w in ins]) for ins in raw_text])

		py = self.model.predict(test_x)
		prob_py = self.model.predict_proba(test_x)
		return py, prob_py

	def fit_vectorizer(self, train_df):
		raw_text = [self.tokenizer.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in train_df["input_ids"]]
		self.vectoriser.fit([" ".join([str(w) for w in ins]) for ins in raw_text])

	def save_model(self, save_path):
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		pickle.dump(self, open(os.path.join(save_path, "model.sav"), "wb"))


class RandomForestSKLearnClassifier(SKLearnClassifier):
	def __init__(self, train_df, max_length=512, **tunable_model_args):
		super().__init__(
			model_class=RandomForestClassifier,
			train_df=train_df,
			max_length=max_length,
			**tunable_model_args
		)


class LogisticRegressionSKLearnClassifier(SKLearnClassifier):
	def __init__(self, train_df, max_length=512, **tunable_model_args):
		super().__init__(
			model_class=LogisticRegression,
			train_df=train_df,
			max_length=max_length,
			**tunable_model_args
		)


class SklearnTokenizer:
	def __init__(self, max_length):
		self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
		self.max_length = max_length

	def tokenize(self, text):
		tokenized_dict = self.tokenizer.encode_plus(
			text=text,
			add_special_tokens=True,
			pad_to_max_length=True,

			max_length=self.max_length,

			return_attention_mask=True,
			truncation=True
		)
		return tokenized_dict['input_ids'], tokenized_dict['attention_mask']
