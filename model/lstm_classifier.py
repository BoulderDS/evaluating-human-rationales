from abc import ABC

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import PreTrainedModel, PretrainedConfig, RobertaModel, RobertaTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput


class LSTMClassifier(PreTrainedModel, ABC):
	def __init__(self, config: PretrainedConfig, embedding_length=768, *inputs, **kwargs):
		super().__init__(config, *inputs, **kwargs)

		self.max_len = config.max_length
		# self.packed_embedding = config.packed_embedding
		self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

		self.embedding = RobertaModel.from_pretrained('roberta-base').embeddings
		# BiLSTM hidden size = 2 times hidden size
		self.lstm = nn.LSTM(
			input_size=embedding_length,
			batch_first=True,
			hidden_size=config.hidden_size,
			num_layers=1,
			dropout=0.2,
			bidirectional=True
		)
		# self.tanh = nn.Tanh()
		self.predictor = nn.Linear(in_features=config.hidden_size * 2, out_features=config.num_labels)
		self.pad_packing = config.pad_packing

	def forward(self, input_ids, labels=None, attention_mask=None):

		result = SequenceClassifierOutput()
		embedding = self.embedding.word_embeddings(input=input_ids)

		if self.pad_packing:
			input_lengths = attention_mask.sum(dim=1)
			packed_embeddings = pack_padded_sequence(embedding, input_lengths, batch_first=True, enforce_sorted=False)
			packed_output, state = self.lstm.forward(packed_embeddings)
		else:
			hidden, state = self.lstm.forward(input=embedding)

		# out = torch.stack([context[0][0], context[0][1]], dim=1)
		out = torch.cat([state[0][0], state[0][1]], dim=1)

		logits = self.predictor(input=out)
		probs = nn.functional.softmax(logits, dim=1)
		py_index = torch.argmax(probs, dim=1)

		if labels is not None:
			result["loss"] = nn.functional.cross_entropy(logits, labels)
		result["logits"] = logits
		result["probs"] = probs
		result["py_index"] = py_index
		return result

	def tokenize(self, text):
		tokenized_dict = self.tokenizer.encode_plus(
			text=text,
			add_special_tokens=True,
			pad_to_max_length=True,
			max_length=self.max_len,
			return_attention_mask=True,
			truncation=True
		)
		return tokenized_dict['input_ids'], tokenized_dict['attention_mask']
