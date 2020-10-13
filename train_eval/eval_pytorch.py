import sklearn.metrics as mt
import numpy as np
import torch


def eval_fn(model, dev_dataloader, nth_epoch, device="cuda"):
	model.eval()
	y_hat = np.zeros(0, dtype=int)
	y = np.zeros(0, dtype=int)
	dev_losses = []
	# for local_batch, local_labels, rationales, attention_masks in dev_dataloader:
	for sample in dev_dataloader:
		with torch.no_grad():
			# sending only a part of the data to GPU
			outputs = model.forward(
				input_ids=sample["input_ids"].to(device),
				labels=sample["labels"].to(device),
				attention_mask=sample["attention_mask"].to(device))
		dev_losses.append(outputs.loss.item())

		y_hat = np.concatenate((y_hat, outputs['py_index'].detach().cpu().numpy()), axis=0)
		y = np.concatenate((y, sample["labels"].numpy()), axis=0)
	dev_acc = mt.accuracy_score(y, y_hat)
	return dev_acc, np.mean(dev_losses)
