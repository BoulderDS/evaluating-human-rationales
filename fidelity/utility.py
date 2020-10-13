import numpy as np
import logging
import torch

logger = logging.getLogger(__name__)


def compute_fidelity(prob_y_hat, prob_y_hat_alpha, fidelity_type="sufficiency", clip=True, dataset_level=False):
	"""
	py_probability = P(y_hat/x)
	py_probability_alpha = P(y_hat/x,alpha)
	clip=True to keep the normalized values between 0 and 1
	clip=False Eraser's definition of sufficiency and comprehensiveness
	dot mean because if it is an array or a list, dataset level fidelity else instance level
	"""
	if clip:
		mean_difference = np.clip(np.array(prob_y_hat) - np.array(prob_y_hat_alpha), 0, 1)
	else:
		mean_difference = prob_y_hat - prob_y_hat_alpha
	if dataset_level:
		mean_difference = np.mean(mean_difference)
	if fidelity_type == "sufficiency":
		return 1 - mean_difference
	else:
		return mean_difference


def itemwize_normalization(fidelity, null_difference, fidelity_type, clip, eps):
	"""
	null_difference = max(0, P(y_hat/x) - P(y_hat/x, 0)); alpha = 0
	fidelity_type = {"sufficiency", "comprehensiveness"}
	clip=True to keep the normalized values between 0 and 1
	clip=False Eraser's definition of sufficiency and comprehensiveness
	Explain denominator
	"""
	# Have a bunch of unit tests to test your functions
	# test if null diff == 1 - suff at 0
	try:
		if fidelity_type == "sufficiency":
			numerator = (fidelity - (1 - null_difference))
			denominator = (1 - (1 - null_difference))
		else:
			numerator = fidelity
			denominator = null_difference

		if np.abs(denominator) < eps:
			result = 0
		else:
			result = numerator / denominator
	except Exception as e:
		logger.info(e)
		return 0

	if clip:
		return np.clip(result, 0, 1)
	else:
		return result


def normalization(fidelity, null_difference, fidelity_type="sufficiency", clip=True, eps=1e-5):
	"""
	Normalize a list of fidelity values given the null difference values
	:param fidelity:
	:param null_difference:
	:param fidelity_type:
	:param clip:
	:param eps:
	:return:
	"""
	normalized_fidelity = []
	for idx, itemwise_fidelity in enumerate(fidelity):
		normalized_fidelity.append(
			itemwize_normalization(
				fidelity=itemwise_fidelity,
				null_difference=null_difference[idx],
				fidelity_type=fidelity_type,
				clip=clip,
				eps=eps
			)
		)

	return normalized_fidelity


def compute_predictions(input_ids,
						model,
						attention_masks=None,
						# device="cuda"
						):
	"""
	Compute the prediction for given input_ids, model and attention masks
	This function returns numpy arrays of labels and probabilities associated with that label
	:param input_ids:
	:param model:
	:param attention_masks:
	:param device:
	:return:
	"""
	if attention_masks is not None:
		result = model.forward(input_ids=input_ids.to(model.device), attention_mask=attention_masks.to(model.device))
	else:
		result = model.forward(input_ids=input_ids.to(model.device))

	# result['probs'] = torch.nn.functional.softmax(result['logits'], dim=1)
	# result['py_index'] = torch.argmax(result['probs'], dim=1)

	return result["py_index"].detach().cpu().numpy(), result["probs"].detach().cpu().numpy()


def compute_null_diff(input_ids, model, predictions, prob_y_hat, tokenizer, return_zero_probs=False):
	"""
	Calculate the null difference
	null difference = max(0, p(y_hat/x) - p(y_hat/x, 0))
	returns the label and prediction probability
	:param return_zero_probs:
	:param prob_y_hat:
	:param predictions:
	:param input_ids:
	:param model:
	:param tokenizer:
	:return:
	"""
	input_ids_reduced, attention_mask_reduced = reduce(input_ids, rationale=None, tokenizer=tokenizer)
	predictions_0, prob_y_hat_0 = compute_predictions(input_ids=input_ids_reduced,
														model=model, attention_masks=attention_mask_reduced)
	prob_y_hat_0_predicted_class = prob_y_hat_0[np.arange(len(prob_y_hat_0)), predictions]
	null_difference = 1 - compute_fidelity(prob_y_hat=prob_y_hat, prob_y_hat_alpha=prob_y_hat_0_predicted_class,
									   fidelity_type="sufficiency")

	if return_zero_probs:
		prob_y_hat_0_strings = []
		for prob in prob_y_hat_0:
			prob_y_hat_0_strings.append(str(prob))
		prob_y_hat_0_strings = np.array(prob_y_hat_0_strings)
		return prob_y_hat_0_strings, null_difference

	return null_difference


def reduce(input_ids, rationale=None, tokenizer=None, fidelity_type="sufficiency"):
	"""
	Reduce the input_ids based on the alpha values or the rationale values
	returns torch tensors of input_ids and attention masks
	:param input_ids:
	:param rationale:
	:param tokenizer:
	:param fidelity_type:
	:return:
	"""
	# unit test this function
	# unit test where you call the model on the reduced data
	# create a model where you can test easily
	# eyeball few example by calling tokenizer input_ids reduced, attention masks reduced,
	# put a breakpoint in the model forward and look at the input ids when you call the reduced prediction
	input_ids_reduced = []
	attention_mask_reduced = []
	for idx in range(len(input_ids)):
		if rationale is None:
			rationale_i = [0]*len(input_ids[idx])
		else:
			rationale_i = rationale[idx]
		input_ids_reduced_i = []
		attention_mask_reduced_i = []
		for j in range(len(input_ids[idx])):
			if input_ids[idx][j] in [0, 2]:
				input_ids_reduced_i.append(input_ids[idx][j].item())
				attention_mask_reduced_i.append(1)
			else:
				if fidelity_type == "sufficiency" and rationale_i[j] >= 0.5:
					input_ids_reduced_i.append(input_ids[idx][j].item())
					attention_mask_reduced_i.append(1)
				elif fidelity_type == "comprehensiveness" and rationale_i[j] < 0.5:
					input_ids_reduced_i.append(input_ids[idx][j].item())
					attention_mask_reduced_i.append(1)
		num_padding_tokens = len(input_ids[idx]) - len(input_ids_reduced_i)
		input_ids_reduced_i = input_ids_reduced_i + [tokenizer.pad_token_id]*num_padding_tokens
		attention_mask_reduced_i = attention_mask_reduced_i + [0]*num_padding_tokens
		input_ids_reduced.append(input_ids_reduced_i)
		attention_mask_reduced.append(attention_mask_reduced_i)
	return torch.tensor(input_ids_reduced), torch.tensor(attention_mask_reduced)
