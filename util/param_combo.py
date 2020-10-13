import itertools


def get_param_combos(input, exclude_combos=None, require_combos=None, verbose=False, add_name=True):
	'''
	Like itertools.product but for dictionaries. Looks for all values that are lists, and returns a dictionary for each unique combination of these value lists.

	For example, will split
	{'a':[100,200], 'b':300}
	into
	[{'name':'a=100','params':{'a':100,'b':300}},
	{'name':'a=200','params':{'a':200,'b':300}}]

	Purpose is mainly for converting a config file into a set of hyperparameter dictionaries to try

	input can also be a list or tuple of dictionaries, in which case it will be combined into one big dictionary, the combinations
	will be calculated, and then the output dictionaries will be split back into tuples according to the keys of the original inputs
	'''

	if type(input) == dict:
		input_dict = input
	elif type(input) == list or type(input) == tuple:
		input_dict = {}
		for sub_dict in input:
			input_dict.update(sub_dict)

	# if not verbose: previous = set_print(False)
	# print('Splitting params')
	anonymous_constants = {k: v for k, v in input_dict.items() if not type(v) == list}
	list_params = {k: v for k, v in input_dict.items() if type(v) == list}

	outputs = []
	excluded_outputs = []
	required_outputs = []
	param_combos = list(product_dict(**list_params))

	print('{} combos'.format(len(param_combos)))

	for unique_param_combo in param_combos:
		name = generate_paramdict_name(unique_param_combo)
		param_combo = {**unique_param_combo, **anonymous_constants}

		if add_name:
			output = {'name': name, 'params': param_combo}
		else:
			output = param_combo
		# Check to see if this param combo needs to be excluded because it violates one of the exclusion or requirement rules
		exclude = False

		if exclude_combos:
			for ed in exclude_combos:
				if dict_contains(param_combo, ed):
					excluded_outputs.append(output)
					exclude = True
					print('Violated exclusion {}'.format(ed))

		# Combo params needs to either fully contain or fully not contain every required combo
		if require_combos:
			for req in require_combos:
				num_present = 0
				num_contained = 0
				for rkey, rval in req.items():
					# if the rkey is not in the dict, we consider the dict to
					if rkey in param_combo:
						num_present += 1
						if param_combo[rkey] == rval:
							num_contained += 1
				if num_contained != num_present and num_contained != 0:
					required_outputs.append(output)
					exclude = True
					print('Violated required combo {}'.format(req))

		if not exclude:
			outputs.append(output)
			print('Including {}'.format(name))
		else:
			print('Excluding {}'.format(name))

	if type(input) == list or type(input) == tuple:
		for output in outputs:
			split_params = []
			for sub_input in input:
				split_params.append({k: v for k, v in output['params'].items() if k in sub_input})
			output['params'] = split_params

	# if not verbose: set_print(previous)

	return outputs


def product_dict(**kwargs):
	'''
	Version of itertools.product that operates on a dictionary of lists to return a list of dictionaries of key-value combinations

	e.g. {a:[1,2],b:[3]} --> [{a:1,b:3},{a:2,b:3}]
	:param kwargs:
	:return:
	'''
	keys = kwargs.keys()
	vals = kwargs.values()
	for instance in itertools.product(*vals):
		yield dict(zip(keys, instance))


def generate_paramdict_name(paramdict, abbreviate_names=True, delimiter='_'):
	if len(paramdict) == 0:
		return 'default'
	else:
		return join_names_and_values(*zip(*sorted(paramdict.items(), key=lambda t: t[0])),
									 abbreviate_names=abbreviate_names, delimiter=delimiter)


def join_names_and_values(names, values, abbreviate_names=True, delimiter='_'):
	if abbreviate_names:
		combo_name = delimiter.join('{}={}'.format(abbreviate(name), value) for name, value in zip(names, values))
	else:
		combo_name = delimiter.join('{}={}'.format(name,value) for name, value in zip(names, values))
	return combo_name


def abbreviate(s, split_token = '_'):
	return ''.join(w[0] for w in s.split(split_token))


def dict_contains(d1, d2):
	'''
	Check to see if dictionary 2 is contained with dictionary 1
	:param d1:
	:param d2:
	:return:
	'''
	contains = True
	for k,v in d2.items():
		if not (k in d1 and d1[k] == v):
			contains = False
			break
	return contains