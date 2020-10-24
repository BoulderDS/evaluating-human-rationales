# Evaluating and Characterizing Human Rationales
This repository contains code for paper https://arxiv.org/abs/2010.04736 appearing in EMNLP2020.

#### Installation
`pip install -r requirements.txt`

#### Structure of the repository
1. `config: ` dictates the experimentation script
    1. `data_config` set the output directory for the experimentation
    2. `data_config` set the appropriate input directories for all the datasets  
    3. `model_config` select among `roberta`, `lstm`, `random forest` or/and `logistic regression` classifier
    4. `data_config` select among `wikiattack`, `sst`, `movie`, `multirc`, `fever` or/and `esnli` datasets
2. `dataset: ` Dataset class with super class `torch.utils.data.Dataset` to create dataloaders and
datasets for trainer
3. `fidelity: ` compute fidelity given predictions or model and input ids (change the code
to use `nlp fidelity`)
4. `model: ` contains classifiers for experimentation
    1. `lstm_classifier` text --> RoBERTa embedding --> BiLSTM --> Linear
    2. `roberta_classifier` robertaForSequenceClassification
    3. `sklearn_classifier` text --> sklearnTokenizer --> sklearnVectoriser
     --> sklearnClassifier(random forest or logistic regression)
5. `plotting: dataset_and_fidelity_analysis_plots` contains the plotting code for all
figures
6. `preliminary_analysis: analyze_datasets and generate_table` to analyze all the
datasets for mean text length, mean rationale length.
7. `scripts: run_experiment_trainer and run_experiment_sklearn` to run experiment on 
roberta classifier, lstm classifier and sklearn classifier respectively.
8. `train_eval: ` contains the code for generating data for fidelity curves and to 
cache prediction to generate plots
9. `util: ` some utility functions

#### Hyperparameters
1. `RoBERTA classifier` 
    1. `hidden_dropout_prob`
2. `LSTM classifier`
    1. `hidden_size`
    2. `pad_packing`
3. `Random Forest classifier`
    1. `n_estimators`
4. `Logistic Regression classifier`
    1. `C`
5. `Training Params`
    1. `learning_rate`
    2. `num_train_epochs`
    3. `weight_decay`
    4. `batch_size`

#### Steps
1. Add the location of the output directory in the config
2. Add the location of the corresponding data directories in the config
3. Choose the model and the data on which analysis is to be performed
4. Run the experiment using `run_experiment_trainer.py` or `run_experiment_sklearn.py`
if the model is lstm or roberta and logistic regression or random forest
respectively.