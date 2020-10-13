import logging

import nlp
import numpy as np

import emnlp20.fidelity.utility as utility

logger = logging.getLogger(__name__)

_CITATION = """
citation is under construction 
"""

_DESCRIPTION = """
description is under construction
"""

_KWARGS_DESCRIPTION = """
kwargs description is also under construction
"""


class Fidelity(nlp.Metric):
	def _info(self):
		return nlp.MetricInfo(description=_DESCRIPTION,
							  citation=_CITATION,
							  inputs_description=_KWARGS_DESCRIPTION,
							  features=nlp.Features({
								  'predictions': nlp.Value('float', id='sequence'),
								  'prob_y_hat': nlp.Value('float', id='sequence'),
								  'prob_y_hat_alpha': nlp.Value('float', id='sequence'),
								  'null_difference': nlp.Value('float', id='sequence'),
								  'model': nlp.Value('float', id='sequence'),
								  'tokenizer': nlp.Value('float', id='sequence'),
								  'mode': nlp.Value('string', id='sequence'),
								  'normalization': nlp.Value('bool', id='sequence'),
							  }))

	def compute(self,
				predictions=None,
				prob_y_hat=None,
				prob_y_hat_alpha=None,
				null_difference=None,
				model=None,
				tokenizer=None,
				input_ids=None,
				alpha=None,
				attention_masks=None,
				fidelity_type="sufficiency",
				clip=True,
				normalization=True,
				device="cpu",
				reduction='mean'):
		"""
		Additional arguments , null_difference=None,
				 model=None, x=None, alpha=None, mode="sufficiency", normalization=True
		alpha is for rationales
		mode is either sufficiency or comprehensiveness
		normalization is True by default
		instance_level or dataset level
		sufficiency(x, y_hat, alpha) = 1 - max(0, P(y_hat/x) - P(y_hat/x, alpha))
		comprehensiveness(x, y_hat, alpha) = max(0, P(y_hat/x) - P(y_hat/x, alpha))
		null difference = max(0, P(y_hat/x) - P(y_hat/x, 0))
		"""
		# if (predictions is None) and \
		# 		(prob_y_hat is None or prob_y_hat_alpha is None or null_difference is None) and \
		# 		(model is None or input_ids is None or alpha is None):
		# 	return "Please provide either predictions or model to compute predictions"

		if prob_y_hat is None:
			predictions, prob_y_hat = utility.compute_predictions(input_ids, model, attention_masks=attention_masks)
			prob_y_hat = prob_y_hat[np.arange(len(prob_y_hat)), predictions]
		if prob_y_hat_alpha is None:
			input_ids_reduced, attention_masks_reduced = utility.reduce(input_ids=input_ids,
																		rationale=alpha,
																		tokenizer=tokenizer,
																		fidelity_type=fidelity_type)

			predictions_alpha, prob_y_hat_alpha = utility.compute_predictions(input_ids=input_ids_reduced,
																			  model=model,
																			  attention_masks=attention_masks_reduced)

			prob_y_hat_alpha = prob_y_hat_alpha[np.arange(len(prob_y_hat_alpha)), predictions]

		fidelity = utility.compute_fidelity(prob_y_hat=prob_y_hat,
											prob_y_hat_alpha=prob_y_hat_alpha,
											fidelity_type=fidelity_type,
											dataset_level=False,
											clip=clip)

		if normalization:
			if null_difference is None:
				null_difference = utility.compute_null_diff(input_ids, model, predictions, prob_y_hat, tokenizer)

			fidelity = utility.normalization(fidelity, null_difference, fidelity_type=fidelity_type, clip=clip)

		if reduction == 'mean':
			return np.mean(fidelity)
		elif reduction is None:
			return fidelity
