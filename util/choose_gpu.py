import subprocess
import re
import os
from pprint import pprint
# import torch

# all_gpus = [0, 1,2,3]

# A line will look something like this: |    3      3395    C   python                                         547MiB |
smi_process_pattern = '\| +(?P<gpu>[0-9]+) +(?P<pid>[0-9]+) +C +\S+ +(?P<mbs>[0-9]+)[a-zA-Z]+ +\|'
gpu_list_pattern = 'GPU (?P<gpu>[0-9]+):.+'

def check_gpu_usage(verbose=True):
	'''
	Runs the nvidia-smi system command and parses the output as a list of GPU numbers that are being used. Also
	looks up each process that is using a GPU and returns some information about it.
	:param verbose:
	:return:
	'''
	smi_str = subprocess.check_output(['nvidia-smi']).decode("utf-8")
	# gpus_in_use = []
	if verbose:
		print('Output of nvidia-smi:')
		print(smi_str)
		print('GPUs process ownership details:')

	gpu_list_str = subprocess.check_output(['nvidia-smi','-L']).decode("utf-8")
	gpu_list_tuples = re.findall(gpu_list_pattern, str(gpu_list_str))
	gpu_utilization = {}
	for gpu in gpu_list_tuples:
		gpu_utilization[int(gpu)] = 0
	# gpu_utilization = {gpu:0 for gpu in all_gpus}


	gpu_process_tuples = re.findall(smi_process_pattern, str(smi_str))
	# gpu_utilization= {gpu:0 for gpu in list(range(torch.cuda.device_count()))}

	for gpu, pid, mbs in gpu_process_tuples:
		# try:
			if int(gpu) not in gpu_utilization:
				gpu_utilization[int(gpu)] = 0

			gpu_utilization[int(gpu)] += int(mbs)
			ps_str = subprocess.check_output('ps -u -p {}'.format(pid),shell=True).decode("utf-8")
			header, info,_ = ps_str.split('\n')
			if verbose:
				print('\tGPU\t{}'.format(header))
				print('\t{}\t{}'.format(gpu, info))
		# except Exception as ex:
		# 	pass

	if verbose:
		print('Total utilization:')
		pprint(gpu_utilization)

	return gpu_utilization

# default_gpus=None, force_default=False, number_of_gpus_wanted=1,
def choose_gpus(verbose=True, max_utilization = 18000):
	'''
	Check which GPUs are in use and choose one or more that are not in use
	:param verbose: whether to print out info about which GPUs are being used
	:param default_gpus: list of default number(s) to go with if it is available
	:param force_default: whether to force the use of the default GPU(s)
	:param number_of_gpus_wanted: whether to force the use of the default GPU(s)

	:return:
	'''
	# smi_str = subprocess.check_output(['nvidia-smi'])
	# gpu_process_tuples = re.findall(pattern, smi_str)

	gpu_utilization = check_gpu_usage().items()

	gpu, utilization = sorted(gpu_utilization, key=lambda t:t[1])[0]

	if utilization <= max_utilization:
		print('Selected GPU {} with {} MBs current utilization.'.format(gpu, utilization))
		return gpu
	else:
		print('Could not find a GPU with less than {} MBs utilization'.format(max_utilization))
		raise Exception



	# if default_gpus is not None and all([default_gpu not in gpus_in_use for default_gpu in default_gpus]):
	# 	if verbose: print('Default GPUs {} are available, so selecting them'.format(default_gpus))
	# 	return default_gpus
	# else:
	# 	if len(all_gpus) - len(gpus_in_use) >= number_of_gpus_wanted:
	# 		chosen_gpus = [gpu for gpu in all_gpus if gpu not in gpus_in_use][:number_of_gpus_wanted]
	# 		if verbose: print('Selected GPUS {} for use'.format(chosen_gpus))
	# 		return chosen_gpus
	# 	else:
	# 		raise Exception('Could not find {} free GPUs. Check output of nvidia-smi command for details.'.format(number_of_gpus_wanted))


def choose_and_set_available_gpus(verbose=True, manual_choice=None,max_utilization=10000):
	'''
	Choose some GPUs and make only them visible to CUDA
	:param verbose:
	:param default_gpu:
	:param force_default:
	:param number_of_gpus_wanted:
	:return:
	'''

	if manual_choice is None:
		chosen_gpu = choose_gpus(verbose,max_utilization=max_utilization)
	else:
		chosen_gpu = int(manual_choice)
	print('Setting GPUs {} as only visible device.'.format(chosen_gpu))
	os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
	return str(chosen_gpu)

# import tensorflow as tf
# def generate_tensorflow_config(lowmem = True):
# 	config = tf.ConfigProto()
# 	config.gpu_options.allow_growth = lowmem
# 	return config

def main():
	print('Checking GPU usage')
	gpus = check_gpu_usage()

	print(gpus)

if __name__ == '__main__':
	main()