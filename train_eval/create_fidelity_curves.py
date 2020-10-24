from train_eval.feature_caching import get_and_save_features
from dataset.dataset import create_test_dataloader
import os

def create_fidelity_curves(
	model=None,
	dataset_path = None,
	dataset_classes=None,
	output_dir=None,
	num_samples=None,
	occlusion_rates=None,
	batch_size=None
):
	'''
	Create information needed for fidelity curves by running model repeatedly over
	dataset with increasingly obscured rationales
	:param model:
	:param dataloader:
	:param output_dir:
	:return:
	'''

	print('Generating fidelity curve')
	num_runs = len(occlusion_rates) * num_samples
	run_num=0
	for occlusion_rate in occlusion_rates:
		for sample_num in range(num_samples):
			run_num +=1
			print(f'Fidelity curve run {run_num}/{num_runs}: occlusion {occlusion_rate} sample {sample_num}')

			occluded_dataloader = create_test_dataloader(
				model=model,
				filepath=dataset_path,
				classes=dataset_classes,
				batch_size=batch_size,
				rationale_occlusion_rate=occlusion_rate
			)

			run_output_dir = os.path.join(output_dir, str(occlusion_rate), str(sample_num))
			get_and_save_features(
				test_dataloader=occluded_dataloader,
				model=model,
				tokenizer=model.tokenizer,
				save_dir=run_output_dir
			)
