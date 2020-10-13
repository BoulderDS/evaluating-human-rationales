from emnlp20.fidelity import fidelity
from emnlp20.fidelity import utility
import numpy as np
import torch


def test_dataset_fidelity(dataloader, model, tokenizer, normalization):
	prob_y_hat = np.array([])
	prob_y_hat_alpha = np.array([])
	null_diff = np.array([])
	for local_batch, local_labels, rationales, attention_masks in dataloader:
		y_hat_i, prob_y_hat_i = utility.compute_predictions(input_ids=local_batch, model=model,
															attention_masks=attention_masks)
		prob_y_hat_i = prob_y_hat_i[np.arange(len(prob_y_hat_i)), y_hat_i]
		prob_y_hat = np.concatenate((prob_y_hat, prob_y_hat_i), axis=0)

		input_ids_red, attention_masks_reduced = utility.reduce(input_ids=local_batch, rationale=rationales,
																tokenizer=tokenizer, fidelity_type="sufficiency")
		y_hat_alpha_i, prob_y_hat_alpha_i = utility.compute_predictions(input_ids=input_ids_red, model=model,
																		attention_masks=attention_masks_reduced)
		prob_y_hat_alpha_i = prob_y_hat_alpha_i[np.arange(len(prob_y_hat_alpha_i)), y_hat_i]
		prob_y_hat_alpha = np.concatenate((prob_y_hat_alpha, prob_y_hat_alpha_i), axis=0)

		null_diff_i = utility.compute_null_diff(input_ids=local_batch, model=model, predictions=y_hat_i,
												prob_y_hat=prob_y_hat_i, tokenizer=tokenizer)
		null_diff = np.concatenate((null_diff, null_diff_i), axis=0)

	f_class = fidelity.Fidelity()
	suff = f_class.compute(prob_y_hat=prob_y_hat, prob_y_hat_alpha=prob_y_hat_alpha, null_difference=null_diff,
						   normalization=normalization)
	comp = f_class.compute(prob_y_hat=prob_y_hat, prob_y_hat_alpha=prob_y_hat_alpha, null_difference=null_diff,
						   fidelity_type="comprehensiveness", normalization=normalization)
	return suff, comp


def test_fidelity(dataloader, model, tokenizer):
	for local_batch, local_labels, rationales, attention_masks in dataloader:
		f_class = fidelity.Fidelity()
		sufficiency = f_class.compute(model=model, tokenizer=tokenizer, input_ids=local_batch, alpha=rationales,
									  attention_masks=attention_masks,
									  fidelity_type="sufficiency")
		comprehensiveness = f_class.compute(model=model, tokenizer=tokenizer, input_ids=local_batch,
											alpha=rationales,
											attention_masks=attention_masks, fidelity_type="comprehensiveness")
		print(sufficiency, comprehensiveness)

		rationales_zero = torch.zeros([32, 256], dtype=torch.float64)
		sufficiency = f_class.compute(model=model, tokenizer=tokenizer, input_ids=local_batch, alpha=rationales_zero,
									  attention_masks=attention_masks, fidelity_type="sufficiency")
		comprehensiveness = f_class.compute(model=model, tokenizer=tokenizer, input_ids=local_batch,
											alpha=rationales_zero,
											attention_masks=attention_masks, fidelity_type="comprehensiveness")
		print(sufficiency, comprehensiveness)
