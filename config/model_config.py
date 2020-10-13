from emnlp20.model.roberta_classifier import RobertaClassifier
from emnlp20.model.lstm_classifier import LSTMClassifier
from emnlp20.model.sklearn_classifier import RandomForestSKLearnClassifier, LogisticRegressionSKLearnClassifier

# model_dict = {'model': ["logistic_regression", "random_forest", "roberta"]}
model_dict = {'model': ["logistic_regression", "random_forest"]}

model_info = {
	'roberta': {
		'class': RobertaClassifier,
		"tunable_model_args": {
			# "hidden_dropout_prob": [0.1, 0.2, 0.3]
			"hidden_dropout_prob": [0.1]
		}
	},
	"lstm": {
		"class": LSTMClassifier,
		"tunable_model_args": {
			"hidden_size": [200],
			"pad_packing": True,
		}
	},
	"random_forest": {
		"class": RandomForestSKLearnClassifier,
		"tunable_model_args": {
			'n_estimators': [4, 16, 64, 256, 512]
		}
	},
	"logistic_regression": {
		"class": LogisticRegressionSKLearnClassifier,
		"tunable_model_args": {
			'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
		}
	}
}
