from transformers import PretrainedConfig
from model.roberta_classifier import RobertaClassifier
import json
import torch
from dataset.dataset import Dataset
import os
import numpy as np
import pandas as pd

load_path = ""
model_load_path = os.path.join(load_path, 'pytorch_model.bin')
with open(os.path.join(load_path, 'config.json'), 'r')as f:
    saved_config = json.load(f)
    saved_config = PretrainedConfig(num_labels=2, **saved_config)

cache_model = RobertaClassifier(config=saved_config)
cache_model.load_state_dict(torch.load(model_load_path))


data_df = pd.read_csv("")

data_df['input_ids'], data_df['attention_mask'] = zip(*data_df['text'].map(cache_model.tokenize))

input_id_tensor = torch.tensor(data_df['input_ids'])
attention_mask_tensor = torch.tensor(data_df['attention_mask'])


labels_tensor = torch.tensor(data_df['classification'].apply(lambda x: int(x)))

dataset_ds = Dataset(input_id_tensor, labels_tensor, attention_mask_tensor, BATCH_SIZE_FLAG=16)
dataloader = torch.utils.data.DataLoader(dataset_ds, batch_size=dataset_ds.BATCH_SIZE_FLAG, shuffle=True)


y_hat = np.zeros(0, dtype=int)
y = np.zeros(0, dtype=int)
cache_model.to("cuda")
for sample in dataloader:
    with torch.no_grad():
# sending only a part of the data to GPU
        outputs = cache_model.forward(
        input_ids=sample["input_ids"].to("cuda"),
        attention_mask=sample["attention_mask"].to("cuda"))
    y_hat = np.concatenate((y_hat, outputs['py_index'].detach().cpu().numpy()), axis=0)
    y = np.concatenate((y, sample["labels"].numpy()), axis=0)
feature_cache_df = pd.DataFrame({
    'true_classes': y,
    'predicted_classes': y_hat,
})



save_dir = "/data/anirudh/output/evaluating_human_rationales/roberta/wikismallred"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
feature_cache_df.to_csv(save_dir + "/feature.csv")