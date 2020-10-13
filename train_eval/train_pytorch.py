from transformers import get_linear_schedule_with_warmup
import numpy as np
import sklearn.metrics as mt
from emnlp20.train_eval.eval_pytorch import eval_fn
import time
import torch
from transformers import Trainer, TrainingArguments, EvalPrediction, PretrainedConfig

class TransformerTrainer():
	'''
	Simple wrapper around a transformers trainer
	'''
	def __init__(self,
				 output_dir=None,
				 save_steps=None,
				 model=None,
				 train_dataset=None,
				 eval_dataset=None,
				 metric_fn=None,
				 do_train=None,
				 **train_args):

		self.do_train=do_train
		self.training_args = TrainingArguments(
			output_dir=output_dir,
			save_steps=save_steps,
			**train_args)

		self.trainer = Trainer(
			model=model,
			args=self.training_args,
			train_dataset=train_dataset,
			eval_dataset=eval_dataset,
			compute_metrics=metric_fn,
			# tokenizer=candidate_model.tokenizer
		)

	def train(self):
		self.trainer.train()


def train_fn(train_dataloader, dev_dataloader, model, optimizer, tokenizer,
			 scheduler, save_dir, n_epochs=3, device="cuda"):
	model.train()
	train_evals = []
	best_dev_acc = 0
	for epoch in range(n_epochs):
		t0 = time.time()
		y_hat = np.zeros(0, dtype=int)
		y = np.zeros(0, dtype=int)
		for batch_idx, (local_batch, local_labels, rationales, attention_masks) in enumerate(train_dataloader):
			elapsed = time.time() - t0
			optimizer.zero_grad()

			outputs = model.forward(
				input_ids=local_batch.to(device),
				labels=local_labels.to(device),
				attention_mask=attention_masks.to(device))
			loss = outputs.loss

			print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.   loss: {:}.'.format(batch_idx, len(train_dataloader), elapsed, loss))

			# measure train accuracy
			y_hat = np.concatenate((y_hat, outputs['py_index'].detach().cpu().numpy()), axis=0)
			y = np.concatenate((y, local_labels.numpy()), axis=0)
			loss.backward()
			optimizer.step()
			scheduler.step()

		dev_evals = eval_fn(model, dev_dataloader, epoch, device=device)
		epoch_evals = {"epoch": epoch, "train_acc": mt.accuracy_score(y, y_hat), "dev_acc": dev_evals[0],
					   "mean_dev_loss": dev_evals[1]}
		print(f'EPOCH: {epoch+1} | Train Accuracy: {epoch_evals["train_acc"]} | Dev accuracy: {epoch_evals["dev_acc"]}'
			  f' | Dev loss: {epoch_evals["mean_dev_loss"]}')
		train_evals.append(epoch_evals)

		# Save model with best dev accuracy
		if epoch_evals["dev_acc"] > best_dev_acc:
			model.save_model(save_dir)
			best_dev_acc = epoch_evals["dev_acc"]
	print("Training Done!")
	return train_evals
