export PYTHONPATH="$PYTHONPATH:/home/anirudh/mita"
#python -u run_experiment.py | tee /data/anirudh/logs/roberta/esnli.txt
#python -u run_experiment_trainer.py | tee /data/anirudh/logs/roberta/esnli_lstm.txt
#python -u run_experiment_trainer.py
#python -u run_experiment.py | tee logs/gta_all_datasets.txt

python -u run_experiment_sklearn.py | tee /data/anirudh/logs/logistic_regression/null_diff.txt