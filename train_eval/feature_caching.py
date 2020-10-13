import numpy as np
import pandas as pd
import os

from emnlp20.fidelity import utility
import os


def get_and_save_features(test_dataloader, model, tokenizer, save_dir, device="cuda"):
	id = np.array(([]))
	prob_y_hat = np.array([])
	prob_y_hat_alpha = np.array([])
	y_hat_alpha = np.array([])
	prob_y_hat_alpha_comp = np.array([])
	y_hat_alpha_comp = np.array([])
	null_diff = np.array([])
	true_classes = np.array([])
	predicted_classes = np.array([])
	zero_probs = np.array([])

	model.to(device)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	for sample in test_dataloader:
		# getting the probabilities
		prob_dict = get_all_probability_values(
			input_ids=sample["input_ids"],
			attention_mask=sample["attention_mask"],
			sufficiency_input_ids=sample["sufficiency_input_ids"],
			sufficiency_attention_mask=sample["sufficiency_attention_mask"],
			comprehensiveness_input_ids=sample["comprehensiveness_input_ids"],
			comprehensiveness_attention_mask=sample["comprehensiveness_attention_mask"],
			model=model,
			tokenizer=tokenizer
			)

		# Appending the probabilities to the numpy arrays
		id = np.concatenate((id, sample["id"]), axis=0)
		prob_y_hat = np.concatenate((prob_y_hat, prob_dict["prob_y_hat_i"]), axis=0)
		prob_y_hat_alpha = np.concatenate((prob_y_hat_alpha, prob_dict["prob_y_hat_alpha_i"]), axis=0)
		y_hat_alpha = np.concatenate((y_hat_alpha, prob_dict["y_hat_alpha_i"]), axis=0)
		prob_y_hat_alpha_comp = np.concatenate((prob_y_hat_alpha_comp, prob_dict["prob_y_hat_alpha_i_comp"]), axis=0)
		y_hat_alpha_comp = np.concatenate((y_hat_alpha_comp, prob_dict["y_hat_alpha_i_comp"]), axis=0)
		null_diff = np.concatenate((null_diff, prob_dict["null_diff_i"]), axis=0)
		true_classes = np.concatenate((true_classes, sample["labels"]), axis=0)
		predicted_classes = np.concatenate((predicted_classes, prob_dict["y_hat_i"]), axis=0)
		zero_probs = np.concatenate((zero_probs, prob_dict["zero_probs"]), axis=0)

	# Creating a dataframe
	feature_cache_df = pd.DataFrame({
		'id': id,
		'prob_y_hat': prob_y_hat,
		'prob_y_hat_alpha': prob_y_hat_alpha,
		'y_hat_alpha': y_hat_alpha,
		'prob_y_hat_alpha_comp': prob_y_hat_alpha_comp,
		# 'y_hat_alpha_comp': y_hat_alpha_comp,
		'null_diff': null_diff,
		'true_classes': true_classes,
		'predicted_classes': predicted_classes,
		'zero_probs': zero_probs
	})

	# Saving the dataframe as a csv file
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	feature_cache_df.to_csv(save_dir + "/feature.csv")


def get_all_probability_values(input_ids, attention_mask, sufficiency_input_ids, sufficiency_attention_mask,
							   comprehensiveness_input_ids, comprehensiveness_attention_mask, model, tokenizer):
	prob_dict = {}

	prob_dict["y_hat_i"], prob_dict["prob_y_hat_i"] = get_prob_for_alpha(
		input_ids=input_ids,
		attention_masks=attention_mask,
		model=model)

	# prob y hat for sufficiency
	prob_dict["y_hat_alpha_i"], prob_dict["prob_y_hat_alpha_i"] = get_prob_for_alpha(
		input_ids=sufficiency_input_ids,
		attention_masks=sufficiency_attention_mask,
		model=model,
		y_hat_prime=prob_dict["y_hat_i"]
	)

	# prob y hat for comprehensiveness
	prob_dict["y_hat_alpha_i_comp"], prob_dict["prob_y_hat_alpha_i_comp"] = get_prob_for_alpha(
		input_ids=comprehensiveness_input_ids,
		attention_masks=comprehensiveness_attention_mask,
		model=model,
		y_hat_prime=prob_dict["y_hat_i"]
	)

	# Null Diff
	prob_dict["zero_probs"], prob_dict["null_diff_i"] = utility.compute_null_diff(
		input_ids=input_ids,
		model=model,
		predictions=prob_dict["y_hat_i"],
		prob_y_hat=prob_dict["prob_y_hat_i"],
		tokenizer=tokenizer,
		return_zero_probs=True
	)

	return prob_dict


def get_prob_for_alpha(input_ids, attention_masks, model, y_hat_prime=None):
	y_hat, prob_y_hat = utility.compute_predictions(
		input_ids=input_ids,
		model=model,
		attention_masks=attention_masks)

	if y_hat_prime is not None:
		eval_prob_y_hat = prob_y_hat[np.arange(len(prob_y_hat)), y_hat]
		prob_y_hat = prob_y_hat[np.arange(len(prob_y_hat)), y_hat_prime]

	else:
		prob_y_hat = prob_y_hat[np.arange(len(prob_y_hat)), y_hat]
	return y_hat, prob_y_hat
