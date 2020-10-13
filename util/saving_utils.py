from shutil import copyfile
import os


def copy_features(load_dir, output_dir):
	copyfile(os.path.join(load_dir, "feature.csv"), os.path.join(output_dir, "feature.csv"))
	copyfile(os.path.join(load_dir, "config.json"), os.path.join(output_dir, "config.json"))
	copyfile(os.path.join(load_dir, "pytorch_model.bin"), os.path.join(output_dir, "pytorch_model.bin"))
	copyfile(os.path.join(load_dir, "training_args.bin"), os.path.join(output_dir, "training_args.bin"))