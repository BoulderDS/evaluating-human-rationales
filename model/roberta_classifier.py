from abc import ABC

import torch
from transformers import RobertaForSequenceClassification, RobertaConfig
from transformers import RobertaTokenizer, RobertaTokenizerFast
from transformers import PreTrainedModel, PretrainedConfig
import math


class RobertaClassifier(PreTrainedModel, ABC):
	config_class = PretrainedConfig

	def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
		super().__init__(config, *inputs, **kwargs)
		self.max_len = config.max_length
		self.model = RobertaForSequenceClassification.from_pretrained(
			'roberta-base',
			num_labels=config.num_labels,
			return_dict=True,
			cache_dir = './transformer_cache'
		)

		self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
		self.config = self.model.config

	def tokenize(self, text):
		if type(text) == float:
			text = ""
		tokenized_dict = self.tokenizer.encode_plus(
			text=text,
			add_special_tokens=True,
			pad_to_max_length=True,
			max_length=self.max_len,
			return_attention_mask=True,
			truncation=True
		)
		return tokenized_dict['input_ids'], tokenized_dict['attention_mask']

	# The `max_len` attribute has been deprecated and will be removed in a future version, use `model_max_length` instead.
	#   FutureWarning,

	def forward(self, input_ids=None, labels=None, attention_mask=None, rationale=None):
		# _, o2 = self.generator(input_ids, attention_masks=attention_masks)
		# assert input_ids[:, 0] == self.tokenizer.cls_token_id * torch.ones_like(input_ids[:, 0])
		result = self.model.forward(
			input_ids=input_ids,
			labels=labels,
			attention_mask=attention_mask
		)
		# take the results off cuda
		# result["logits"] = result["logits"].detach().cpu()
		result['probs'] = torch.nn.functional.softmax(result['logits'], dim=1)
		result['py_index'] = torch.argmax(result['probs'], dim=1)
		return result
