import pandas as pd
pd.set_option("display.precision", 1)

data_df = pd.read_csv("")
corr_dataset_dict = {}
corr_dataset_dict["Wikipedia personal attacks"] = {"abbv": "WikiAttack", "Task type": "Cls", "Granularity": "Token", "Comprehensive": "CHECKMARK", "Class asymmetry": "CHECKMARK"}
corr_dataset_dict["Stanford treebank"] = {"abbv": "SST", "Task type": "Cls", "Granularity": "Token", "Comprehensive": "CHECKMARK", "Class asymmetry": "CROSSMARK"}
corr_dataset_dict["Movie reviews"] = {"abbv": "Movie", "Task type": "Cls", "Granularity": "Token", "Comprehensive": "CROSSMARK", "Class asymmetry": "CROSSMARK"}
corr_dataset_dict["MultiRC"] = {"abbv": "MultiRC", "Task type": "RC", "Granularity": "Sentence", "Comprehensive": "CHECKMARK", "Class asymmetry": "CROSSMARK"}
corr_dataset_dict["FEVER"] = {"abbv": "FEVER", "Task type": "RC", "Granularity": "Sentence", "Comprehensive": "CROSSMARK", "Class asymmetry": "CROSSMARK"}
corr_dataset_dict["E-SNLI"] = {"abbv": "E-SNLI", "Task type": "RC", "Granularity": "Token", "Comprehensive": "CHECKMARK", "Class asymmetry": "CHECKMARK"}


data_df["Task type"] = data_df["dataset"].apply(lambda s: corr_dataset_dict[s]['Task type'])
data_df["Granularity"] = data_df["dataset"].apply(lambda s: corr_dataset_dict[s]['Granularity'])
data_df["Comprehensive"] = data_df["dataset"].apply(lambda s: corr_dataset_dict[s]['Comprehensive'])
data_df["Class asymmetry"] = data_df["dataset"].apply(lambda s: corr_dataset_dict[s]['Class asymmetry'])
data_df["mean_rationale_percent"] = data_df["mean_rationale_percent"].apply(lambda s: 100*s)

data_df["dataset"] = data_df["dataset"].apply(lambda s: corr_dataset_dict[s]['abbv'])
data_df = data_df[["dataset", "mean_text_length", "Task type", "mean_rationale_length", "mean_rationale_percent", "Comprehensive",  "Granularity", "Class asymmetry"]]
data_df.columns = ["Dataset", "Text length", "Task type", "Rationale length", "Ratio", "Comprehensive", "Granularity", "Class asymmetry"]
print(data_df)
print(data_df.to_latex(index=False))
print("Done!")
