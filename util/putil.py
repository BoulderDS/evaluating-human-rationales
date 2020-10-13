import inspect
import itertools
import json
import math
import os
import re
import smtplib
import string
# import intervaltree as it
import sys
import traceback
from collections import OrderedDict
# from bs4 import BeautifulSoup
from copy import copy
from datetime import datetime
from email.mime.text import MIMEText

import nltk
import numpy as np
import pandas
import pandas as pd
import pytz
from scipy import stats

# from modeling import rationale as r
# from modeling import rationale as r
# from modeling.train_and_evaluate_models import seed

pd.set_option('display.width', 3000)
pd.set_option('max_colwidth', 50)
pd.set_option('max_columns', 20)
pd.set_option('precision', 3)

np.set_printoptions(threshold=10000)

'''
A giant undifferentiated grab-bag of utility functions
'''


# Columns that should be included in a processed data file


def year2datetime(year):
	return datetime(year=year, month=1, day=1)


def replace_wiki_tokens(s):
	'''
	Replace tokens specific to the Wikipedia data with correct characters
	:param s:
	:return:
	'''
	return s.replace('TAB_TOKEN', '\t').replace('NEWLINE_TOKEN', '\n')


def replace_wiki_tokens_with_placeholders(s):
	'''
	Replace tokens specific to the Wikipedia data with correct characters
	:param s:
	:return:
	'''
	return s.replace('TAB_TOKEN', ' ').replace('NEWLINE_TOKEN', ' ').strip()


tokenizer = nltk.tokenize.WordPunctTokenizer()


def process_text_to_pd(s, unicode=False):
	return pandas.Series(process_text_to_strings(s, unicode))


def process_text_to_strings(s, unicode=False, ptokenizer=tokenizer):
	'''
	Process a text for modeling. Return both a processed version of the text and a description
	of how it was tokenized.
	:param s:
	:return:
	'''

	tokens, spans = process_text(s, unicode, ptokenizer=ptokenizer)

	span_text = json.dumps(spans)
	processed_text = ' '.join(tokens)

	str(processed_text)
	return processed_text, span_text


def process_text(s, unicode=False, ptokenizer=tokenizer):
	'''
	Process a text for modeling. Return both a processed version of the text and a description
	of how it was tokenized.
	:param s:
	:return:
	'''

	if not unicode:
		processed = s.lower().strip()
	else:
		processed = s.decode('utf-8').lower().strip()

	spans = list(ptokenizer.span_tokenize(processed))
	tokens = ptokenizer.tokenize(processed)
	return tokens, spans


def analyze_column(srs):
	counts, bins = np.histogram(srs, bins=5)
	print('Distribution of values:')
	print('Bins: {}'.format(bins))
	print('Values: {}'.format(counts))
	print('Fractions: {}'.format(counts / float(srs.shape[0])))


def wiki_revid_to_url(rev_id):
	'''
	Generate a URL for a Wikipedia revision ID
	:param rev_id:
	:return:
	'''
	return 'https://en.wikipedia.org/w/index.php?oldid={}'.format(rev_id)


def reddit_id_to_longint(id):
	'''
	Turns out reddit IDs are base-36, so conversion is easy
	:param id:
	:return:
	'''
	return long(id, 36)


def reddit_id_to_url(id, subreddit):
	return 'https://www.reddit.com/r/{}/comments/{}'.format(subreddit, id)


def reddit_utc_to_datetime(utc):
	return datetime.fromtimestamp(utc)


def strip_html(s):
	d = BeautifulSoup(s, 'html')
	return d.text


def guardian_time_to_datetime(s):
	'''
	E.g. 25 Nov 2013 16:29
	:param s:
	:return:
	'''
	try:
		return datetime.strptime(s, '%d %b %Y %H:%M')
	except:
		return None


def guardian_url(row):
	'''
	Create a permalink to a guardian comment from a series consisting of the article URL and the comment ID
	:param srs:
	:return:
	'''
	return row['article'] + '#comment-' + str(row.name)


def generate_iac_id(row):
	'''
	Generate a numeric ID for a 4forums IAC post. This is dumb, but the dataset doesn't give a
	better unique numeric ID for these comments
	:param row:
	:return:
	'''
	try:
		b1 = bin(int(row['page_id']))[2:]
		b1 = '0' * (16 - len(b1)) + b1

		b2 = bin(int(row['tab_number']))[2:]
		b2 = '0' * (16 - len(b2)) + b2

		return long(b1 + b2, 2)
	except:
		try:
			b1 = bin(int(row['page_id.1']))[2:]
			b1 = '0' * (16 - len(b1)) + b1

			b2 = bin(int(row['tab_number.1']))[2:]
			b2 = '0' * (16 - len(b2)) + b2

			return long(b1 + b2, 2)
		except:
			return None


def int_or_none(x):
	try:
		return int(x)
	except:
		return None


def split_params(params, abbreviate_names=True, delimiter='_', only_multiobject_lsts=False, exclude_combos=None,
				 require_combos=None):
	'''
	Splits a dictionary of parameters where some values are lists, into one set of parameters per combination of listed elements. Gives each one a name.
	:param params: dictionary
	:return: param_sets: a list of tuples. First element of each tuple is a string name for that param set. Second element is a dictionary constituting the set.
	'''

	param_sets = []

	namelsts = [(name, lst) for name, lst in params.items() if
				type(lst) == list and (not only_multiobject_lsts or len(lst) > 1)]

	if len(namelsts) == 0:
		singleton_params = {k: v[0] if type(v) == list and len(v) == 1 else v for k, v in params.items()}
		return [('all_params', singleton_params, singleton_params)], 0

	namelsts = sorted(namelsts, key=lambda x: x[0])

	names, lsts = zip(*namelsts)

	consistent_params = {k: v for k, v in params.items() if k not in names}
	consistent_params.update({k: v[0] for k, v in consistent_params.items() if
							  type(v) == list and len(v) == 1})  # Unpack any singleton list values

	combinations = list(itertools.product(*lsts))

	excluded = 0
	for values in combinations:

		if abbreviate_names:
			combo_name = delimiter.join('{}={}'.format(abbreviate(name), value) for name, value in zip(names, values))
		else:
			combo_name = delimiter.join('{}={}'.format(name, value) for name, value in zip(names, values))

		unique_params = {name: value for name, value in zip(names, values)}
		combo_params = consistent_params.copy()
		combo_params.update(unique_params)

		exclude = False

		if exclude_combos:
			for ed in exclude_combos:
				if dcontains(combo_params, ed):
					exclude = True

		# Combo params needs to either fully contain or fully not contain every required combo
		if require_combos:
			for req in require_combos:
				num_contained = 0
				for rkey, rval in req.items():
					if combo_params[rkey] == rval:
						num_contained += 1
				if num_contained != len(req) and num_contained != 0:
					exclude = True

		if not exclude:
			param_sets.append((combo_name, combo_params, unique_params))
		else:
			excluded += 1

		pass

	return param_sets, excluded


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


def split_params_2(params, secondary_params=None, named_constants=None, exclude_combos=None, require_combos=None,
				   abbreviate_names=True, delimiter='_', verbose=False):
	'''
	This function takes in one or more dictionaries of hyperparameters and calculates all combinations thereof in accordance
	with optional rules about combinations that cannot occur together (exclude_combos) or must occur together (require_combos).

	Generally speaking, the primary params are set to be the hyperparams where we want to generate and save one model per combination (e.g. attention style),
	while the secondary params are set to be those for which we want to scan across several trained models and save only the best (e.g. hidden size).

	example:
	split_params_2({'a':[1,2],'b':[3,4],'c':5},
		secondary_params={'d':[6,7],'e':8},
		exclude_combos=[{'a':1,'b':3}], #We never want a to be 1 when b is 3 and vice versa
		require_combos=[{'a':2,'d':7}]) #We always want a to be 2 when d is 7 and vice versa

	-->

	[{'name': 'a=1_b=4',
	  'params': {'a': 1, 'b': 4, 'c': 5},
	  'sub_params': [{'name': 'a=1_b=4_d=6',
					  'params': {'a': 1, 'b': 4, 'c': 5, 'd': 6, 'e': 8},
					  'unique_name': 'd=6'}],
	  'unique_name': 'a=1_b=4'},

	 {'name': 'a=2_b=3',
	  'params': {'a': 2, 'b': 3, 'c': 5},
	  'sub_params': [{'name': 'a=2_b=3_d=7',
					  'params': {'a': 2, 'b': 3, 'c': 5, 'd': 7, 'e': 8},
					  'unique_name': 'd=7'}],
	  'unique_name': 'a=2_b=3'},

	 {'name': 'a=2_b=4',
	  'params': {'a': 2, 'b': 4, 'c': 5},
	  'sub_params': [{'name': 'a=2_b=4_d=7',
					  'params': {'a': 2, 'b': 4, 'c': 5, 'd': 7, 'e': 8},
					  'unique_name': 'd=7'}],
	  'unique_name': 'a=2_b=4'}]

	:param params: dictionary of hyperparam values. This function will split on any list-valued hyperparam
	:param secondary_params: dictionary of hyperparam values. The function will sub-split based on these values and store those combinations in an inner level of the eventual function output
	:param named_constants: dictionary of hyperparams whose values will stay constant, but which we nevertheless want to end up in the combo names
	:param exclude_combos: list of dictionaries indicating hyperparameter value combinations that cannot occur together
	:param require_combos: list of dictionaries indicating hyperparameter combinations that must occur together if they occur at all
	:param abbreviate_names: whether to abbreviate names in the combo name (e.g. sparsity_loss_weight --> slw)
	:param delimiter: delimiter of key-value equality statements in combo name
	:param verbose:
	:return: list of dictionaries with following structure:
	'name':string representation of the param combo
	'params':dictionary of param values
	'sub_params':list of dictionaries with same structure, with parent params set as a named combo
	'unique_name':string representation of only the params whose values varied (i.e. were specified as lists in the input)

	'''

	if not verbose: previous = ioff()
	iprint('Splitting params')
	anonymous_constants = {k: v for k, v in params.items() if not type(v) == list}
	list_params = {k: v for k, v in params.items() if type(v) == list}

	outputs = []
	excluded_outputs = []
	required_outputs = []
	param_combos = list(product_dict(**list_params))

	iprint('{} combos'.format(len(param_combos)))

	for param_combo in param_combos:
		unique_name = generate_paramdict_name(param_combo)
		if named_constants is not None:
			param_combo.update(named_constants)
		named_combo = param_combo.copy()
		name = generate_paramdict_name(named_combo)

		if verbose: iprint(name)
		param_combo.update(anonymous_constants)

		iinc()
		output = {'name': name, 'unique_name': unique_name, 'params': param_combo}

		# Check to see if this param combo needs to be excluded because it violates one of the exclusion or requirement rules
		exclude = False

		if exclude_combos:
			for ed in exclude_combos:
				if dcontains(param_combo, ed):
					excluded_outputs.append(output)
					exclude = True
					if verbose: iprint('Violated exclusion {}'.format(ed))

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
					if verbose: iprint('Violated required combo {}'.format(req))

		if not exclude:
			outputs.append(output)
			# If secondary params is present, recursively run this function on that dictionary, passing in the primary params dict as a set of constants
			if secondary_params is not None:
				all_params = secondary_params.copy()
				all_params.update(param_combo)
				secondary_outputs = split_params_2(all_params, secondary_params=None, named_constants=named_combo,
												   exclude_combos=exclude_combos, require_combos=require_combos,
												   abbreviate_names=abbreviate_names, delimiter=delimiter,
												   verbose=verbose)
				output['sub_params'] = secondary_outputs

			if verbose: iprint('Including {}'.format(name))
		else:
			if verbose: iprint('Excluding {}'.format(name))
			pass

		idec()
	if not verbose: iset(previous)

	return outputs


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
		combo_name = delimiter.join('{}={}'.format(name, value) for name, value in zip(names, values))
	return combo_name


# def test_split_params():
# 	iprint('Testing split_params function')
# 	#Input
# 	primary_params = {
# 		'a':100,
# 		'b':[200],
# 		'c':[300,400],
# 		'd':[500,600]
# 	}
# 	secondary_params={
# 		'e':700,
# 		'f':[800],
# 		'g':[900,1000],
# 		'h':[1100, 1200]
# 	}
# 	exclude_combos = [
# 		{'c':300, 'd':500}, # 1) If c is 300, d cannot be 500 and vice versa
# 		{'d':600, 'g':900} # 2) If d is 600, g cannot be 900 and vice versa
# 	]
# 	require_combos = [
# 		{'c':400,'h':1200}, # 3) if c is 400, h must be 1200 and vice versa
# 		{'g':900,'h':1100} # 4) if g is 900, h must be 1100 and vice versa
# 	]
#
# 	#output
# 	named_constants = { 'b':200, 'f':800}
# 	anonymous_constants = {'a':100,'e':700}
# 	combo_collections = [
# 		{'params':{'b':200,'c':300,'d':500},
# 		 'sub_params':[
# 			# {'g':900, 'h':1100}, # 1)
# 			# {'g':900, 'h':1200},# 1)
# 			# {'g':1000, 'h':1100}, # 1)
# 			# {'g':1000, 'h':1200} #1)
# 		]}, # 1)
# 		{'params':{'b':200,'c':300,'d':600},
# 		 'sub_params':[
# 			# {'g':900, 'h':1100}, # 2)
# 			# {'g':900, 'h':1200}, # 2)
# 			# {'g':1000, 'h':1100}, #4)
# 			# {'g':1000, 'h':1200} #3)
# 		]},
# 		{'params':{'b':200,'c':400,'d':500},
# 		 'sub_params':[
# 			# {'g':900, 'h':1100}, # 3)
# 			# {'g':900, 'h':1200}, # 4)
# 			# {'g':1000, 'h':1100}, # 3)
# 			{'g':1000, 'h':1200}
# 		]},
# 		{'params':{'b':200,'c':400,'d':600},
# 		 'sub_params':[
# 			# {'g':900, 'h':1100}, # 2)
# 			# {'g':900, 'h':1200}, # 2)
# 			# {'g':1000, 'h':1100}, # 3)
# 			{'g':1000, 'h':1200}
# 		]},
# 	]
#
# 	for combo_collection in combo_collections:
# 		combo_collection.
# 		combo_collection['name'] = join_names_and_values(*zip(*sorted(combo_collection['constants'].items(), key=lambda t:t[0])))
# 		new_combos = []
# 		for combo in combo_collection['combos']:
# 			new_combo = combo_collection['constants']
# 			new_combo.update(combo)
# 			new_combo.update(constant_values)
# 			name = generate_paramdict_name(new_combo)
# 			new_combos.append({'name':name, 'params':new_combo})
# 		combo_collection['combos'] = new_combos
# 		del(combo_collection['constants'])
#
# 	function_output = split_params_2(primary_params, secondary_params=secondary_params, exclude_combos=exclude_combos, require_combos=require_combos)
#
# 	pprint(function_output)
# 	pprint(combo_collections)
#
# 	assert(function_output == combo_collections)
#
# 	pass


def abbreviate(s, split_token='_'):
	return ''.join(w[0] for w in s.split(split_token))


def dcontains(d1, d2):
	'''
	Check to see if dictionary 2 is contained with dictionary 1
	:param d1:
	:param d2:
	:return:
	'''
	contains = True
	for k, v in d2.items():
		if not (k in d1 and d1[k] == v):
			contains = False
			break
	return contains


ect = pytz.timezone('US/Eastern')


# ect = pytz.timezone('US/Mountain')

def now():
	now = datetime.now(ect)
	# formatted = now.strftime("%Y-%m-%d %I:%M %p %z")
	# now.strftime = lambda self, format:formatted
	return now


def today():
	now = datetime.now(ect)
	today = now.date()
	return today


'''
A bunch of convenience functions for printing at various levels of indendation
'''
iprint_tab_level = 0
iprinting_on = True


def iprint(o='', *args, inc=0, log_info=True, end='\n', sep='|   '):
	if len(str(o)) > 10000:
		print('')

	if iprinting_on:
		msg = sep * (iprint_tab_level + inc) + str(o).replace('\n', '\n' + sep * (iprint_tab_level + inc))
		if log_info: msg = msg + ' <{}>'.format(fdatetime(now()))
		print(msg, *args, end=end)


def lprint(*args, **kwargs):
	iprint(log_info=True, *args, **kwargs)


def itoggle():
	global iprinting_on
	iprinting_on = not iprinting_on


def ioff():
	global iprinting_on
	previous = iprinting_on
	iprinting_on = False
	return previous


def ion():
	global iprinting_on
	previous = iprinting_on

	iprinting_on = True
	return previous


def iget():
	return iprint_tab_level


def iset(b):
	if type(b) == bool:
		global iprinting_on
		iprinting_on = b
	elif type(b) == int:
		global iprint_tab_level
		iprint_tab_level = b


increment_levels = []


def iinc(key=None, max=20):
	global iprint_tab_level
	global increment_levels

	if iprint_tab_level >= max:
		print(f'WARNING:iprint tab level exceeding max of {max}')
	else:
		iprint_tab_level += 1

	try:
		increment_levels.append(inspect.stack()[1].function)
	except:
		pass


# if key != None:
# 	increment_levels.append(key)

def idec(key=None):
	global iprint_tab_level
	global increment_levels

	iprint_tab_level = max(iprint_tab_level - 1, 0)

	# if key is not None:
	# 	oldkey = increment_levels.pop()
	# 	if oldkey != key:
	# 		iprint('WARNING: decrement key "{}" does not match last increment key {}'.format(oldkey, key))

	try:
		oldkey = increment_levels.pop()

		stack = inspect.stack()
		if len(stack) > 1:
			newkey = stack[1].function
			if oldkey != newkey:
				iprint('WARNING: decrement key "{}" does not match last increment key "{}"'.format(newkey, oldkey))
			# print('parp')
	except:
		pass


ticks = {}


def tick(key='', verbose=True):
	current_time = now()

	if verbose:
		if key != '':
			iprint('Tick. {} | {}'.format(ftime(current_time), key))
		else:
			iprint('Tick. {}'.format(ftime(current_time)))

	ticks[key] = current_time


def tock(key='', comment=None, verbose=True):
	'''
	Convenience function for printing a timestamp with a comment
	:param comment:

	:return:
	'''
	last_tick = ticks[key]

	current_tick = now()

	ps = 'Tock. {}'.format(ftime(current_tick))

	ps += ' | {} elapsed.'.format(finterval(current_tick - last_tick))

	if key != '':
		ps += ' | {}'.format(key)

	if verbose:
		iprint(ps)
	return current_tick - last_tick


def ftime(dt):
	return dt.strftime("%I:%M %p")


def fdatetime(dt):
	return dt.strftime("%I:%M %p %m/%d/%Y")


def fdatetime_s(dt):
	return dt.strftime("%I:%M:%S %p %m/%d/%Y")


def rdatetime(dtstr):
	return datetime.strptime(dtstr, "%I:%M %p %m/%d/%Y")


def rdatetime2(dtstr):
	return datetime.strptime(dtstr, "%I:%M:%S %p %m/%d/%Y")


def rdatetime3(dtstr):
	return datetime.strptime(dtstr, "%m/%d/%Y %I:%M")


def rdatetime4(dtstr):
	return datetime.strptime(dtstr, "%m/%d/%Y, %I:%M:%S %p")


def rdatetime5(dtstr):
	return datetime.strptime(dtstr, "%d/%m/%Y, %H:%M:%S")


def rdatetime6(dtstr):
	return datetime.strptime(dtstr, "%Y/%m/%d %H:%M:%S")


def rdatetime7(dtstr):
	return datetime.strptime(dtstr, "%d/%m/%Y %H:%M:%S")


def rdatetime8(dtstr):
	return datetime.strptime(dtstr, "%m/%d/%Y %I:%M:%S %p")


def fdatetime_file(dt):
	return dt.strftime("%I.%M_%p_%m-%d-%Y")


def finterval(interval):
	return str(interval)


def remove(item, sequence):
	new_sequence = copy(sequence)
	try:
		new_sequence.remove(item)
	except Exception as x:
		iprint('Warning: could not remove item {} from sequence {}. Error message: {}'.format(item, shorten(sequence),
																							  x.message))
	return new_sequence


def shorten(o, max_len=1000, double_sided=True):
	s = str(o)
	if len(s) > max_len:
		if double_sided:
			s = s[0:max_len / 2] + '...' + s[-max_len / 2:]
		else:
			s = s[0:max_len] + '...'

	return s


numbered_filename_pattern = re.compile('([0-9\.]+)_.+')


def highest_current_file_prefix(directory):
	files = os.listdir(directory)
	numbers = []
	for file in files:
		match = re.match(numbered_filename_pattern, file)
		if match:
			numbers.append(float(match.groups()[0]))
	if len(numbers) > 0:
		return max(numbers)
	else:
		return None


def rationale_to_annotation(rationale, tokenization=None):
	annotations = []
	r = 0
	last = None
	current = None
	start = None
	end = None
	while r < len(rationale):
		current = rationale[r]
		if rationale[r] == 1:
			if last == 1:
				end = tokenization[r][1] if tokenization is not None else r + 1
			else:
				start, end = tokenization[r] if tokenization is not None else (r, r + 1)

		else:
			if last == 1:
				annotations.append((start, end))
			else:
				pass

		last = current
		r += 1

	if last == 1:
		annotations.append((start, end))

	return annotations


def rationales_to_annotation(rationale1, rationale2, r1_name, r2_name, tokenization):
	'''
	Turn two competing rationales into an annotation
	:param rationale1:
	:param rationale2:
	:param r1_name:
	:param r2_name:
	:param tokenization:
	:return:
	'''
	try:
		assert (len(rationale1) == len(rationale2))
	except:
		pass

	combined_rationale = []
	for i in range(len(rationale1)):
		r1i = rationale1[i]
		r2i = rationale2[i]
		if r1i == 1 and r2i == 1:
			combined_rationale.append(1)
		elif r1i == 1 and r2i == 0:
			combined_rationale.append(2)
		elif r1i == 0 and r2i == 1:
			combined_rationale.append(3)
		else:
			combined_rationale.append(0)

	labels = [None, '{}_and_{}'.format(r1_name, r2_name), '{}_only'.format(r1_name), '{}_only'.format(r2_name)]

	annotations = []
	r = 1
	last = combined_rationale[0]
	current = None
	start = tokenization[0][0]
	end = tokenization[0][1]
	while r < len(combined_rationale):
		current = combined_rationale[r]
		if current == last:  # if the current value is the same as the last, just extend the current annotation
			end = tokenization[r][1]
		else:  # otherwise, we've come to the end of one annotation and the beginning of a new one
			# deal with old one
			if last == 0:  # If the values for the old one are 0, skip it
				pass
			else:  # otherwise add an annotation with the appropriate label
				annotations.append((labels[last], start, end))

			# deal with new one
			start = tokenization[r][0]
			end = tokenization[r][1]

		last = current
		r += 1

	# Deal with the last annotation
	if last == 0:  # If the values for the old one are 0, skip it
		pass
	else:  # otherwise add an annotation with the appropriate label
		annotations.append((labels[last], start, end))

	return annotations


def test_rationale_to_annotation():
	print('Testing rationale_to_annotation function()')
	rationale = [1, 0, 0, 1, 1, 0]

	tokenization = [[0, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, 30]]

	desired_annotations = [(0, 5), (16, 25)]

	assert (rationale_to_annotation(rationale, tokenization) == desired_annotations)

	print('No problems found with rationale_to_annotation function')


# test_rationale_to_annotation()


def annotations_to_rationales(annotation_dict, tokenization):
	'''
	Convert a set of annotations to a set of rationales. Currently using the rule that if part of a token is included, then the whole token is included in the rationale.

	Possible modification would be to see if a majority of the token is included, and only include it in that case. I don't think it will make much difference.

	:param annotation_dict: annotator num --> list of annotation tuples of form (label, relative_span_start, relative_span_end)
	:param tokenization: list of spans defining how the original text was tokenized
	:return: rationale_dict: annotator_num --> single rationale, a list of 0s and 1s the length of the tokenization
	'''

	rationale_dict = {}
	ttree = it.IntervalTree()
	for i, (tstart, tend) in enumerate(tokenization):
		ttree.addi(tstart, tend, data=i)

	for annotator_num, annotation_list in annotation_dict.items():
		rationale_dict[annotator_num] = annotation_to_rationale(annotation_list, tokenization, ttree)

	return rationale_dict


def annotation_to_rationale(annotation_list, tokenization, ttree=None, tokenwise_annotations=False, vector_type=None):
	'''

	:param annotation_list: A list of tuples of the form (label, start, end) or (start, end)
	:param tokenization: A list of character spans which define the tokenization of some text
	:param ttree: An interval tree, if we are running this function on the same text over and over again
	:param tokenwise_annotation: If this is false, then the annotations in annotation_list are character spans. If true, they are token spans.
	:return:
	'''
	rationale = [0 for span in tokenization]

	if not tokenwise_annotations:
		if not ttree:
			ttree = it.IntervalTree()
			for i, (tstart, tend) in enumerate(tokenization):
				ttree.addi(tstart, tend, data=i)

		for annotation_tuple in annotation_list:
			if len(annotation_tuple) == 3:
				label, astart, aend = annotation_tuple
			else:
				astart, aend = annotation_tuple
			for token in ttree.search(astart, aend):
				rationale[token.data] = 1
	else:
		for annotation_tuple in annotation_list:
			astart, aend = annotation_tuple
			for i in range(astart, aend):
				rationale[i] = 1

	if vector_type:
		rationale = np.array(rationale, dtype=vector_type)

	return rationale


def split_list(lst, num):
	'''
	Split a list into num equal-ish sized chunks
	:param lst:
	:param num:
	:return:
	'''

	lsts = [[] for i in range(num)]
	for i in range(len(lst)):
		lsts[(i * num) / len(lst)].append(lst[i])

	lsts = [x for x in lsts if len(x) > 0]

	return lsts


def safe_mean(sequence):
	return np.mean([x for x in sequence if not np.isnan(x)])


def bound(v, minv=0, maxv=1):
	return min(maxv, max(minv, v))


def mean_dict_list(dct_lst, nan_safe=False, prefix=None):
	collected = {}
	for dct in dct_lst:
		for k, v in dct.items():
			if not prefix:
				pk = k
			else:
				pk = prefix + k

			try:
				fv = float(v)
				if not pk in collected:
					collected[pk] = []
				collected[pk].append(fv)
			except:
				pass

	if not nan_safe:
		mean = {k: np.mean(v) for k, v in collected.items()}
	else:
		mean = {k: safe_mean(v) for k, v in collected.items()}

	return mean


class Logger():
	def __init__(self, logfile):
		self.terminal = sys.stdout
		self.log = open(logfile, 'a')
		self.filename = logfile

	def write(self, message):
		self.terminal.write(message)
		# if self.log_dates:
		# 	self.log.write('<{} - {} - {}>\n'.format(sys.argv[0].split('/')[-1], __file__.split('/')[-1], ftime(now())))
		self.log.write(message)
		self.flush()

	def flush(self):
		self.terminal.flush()
		self.log.flush()

	def close(self):
		self.log.close()

	def clear(self):
		self.log.close()
		open(self.filename, 'w').close()
		self.log = open(self.filename, 'a')


def symlink(src, dest, replace=False):
	if replace and os.path.exists(dest):
		iprint('Replacing existing symlink with new one at {}'.format(dest))
		os.remove(dest)
	os.symlink(src, dest)


def convert_to_distribution(y, output_distribution_interpretation):
	'''
	Convert an nx1 vector of target values to an nxm matrix of target distributions. If the interpreation is "regression", then create one-hot vectors with a 1 in the appropriate bucket.

	If the interpretation is "class_probability", then create 2d vectors with binary class probabilities
	:param y:
	:param output_distribution_size:
	:param output_distribution_interpretation:
	:return:
	'''

	dy = np.empty((len(y), 2))
	for i, yi in enumerate(y):
		try:
			dy[i] = convert_item_to_distribution(yi, output_distribution_interpretation)
		except:
			raise

	# mean_distribution += dy[i]

	mean = np.mean(y, axis=0)

	if output_distribution_interpretation == 'class_probability':
		std = np.sqrt(mean * (1 - mean))
	else:
		std = np.std(y, axis=0)

	mean_distribution = np.concatenate([mean, std])

	return dy, mean_distribution


def convert_item_to_distribution(yi, output_distribution_interpretation, minimum_probability=0.01, sigma=0.1):
	if output_distribution_interpretation == 'class_probability':
		# if output_distribution_size != 2:
		# 	raise Exception('ERROR: Cannot interpret a target value as class probability for any other than two classes')
		#
		# yi = max(minimum_probability, min(1-minimum_probability, float(yi)))
		#
		# dyi = [1-yi, yi]
		dyi = [float(yi), float(max(sigma, np.sqrt(yi * (1 - yi))))]
	elif output_distribution_interpretation == 'one_hot':
		# # interv
		# # dyi = [minimum_probability]*output_distribution_size
		# #
		# # dyi[min(int(np.floor(yi*output_distribution_size)),output_distribution_size-1)]=1.0-(output_distribution_size-1)*minimum_probability
		#
		# interval = 1.0 / output_distribution_size
		# bdy = list(frange(interval / 2, 1.0, interval))
		dyi = [yi, sigma]

	return np.matrix(dyi, dtype='float32')


def convert_from_distribution(dy, output_distribution_interpretation):
	y = np.empty((len(dy), 1))
	for i, dyi in enumerate(dy):
		y[i] = convert_item_from_distribution(dyi, output_distribution_interpretation)

	return y


def convert_item_from_distribution(dyi, output_distribution_interpretation):
	# if output_distribution_interpretation == 'class_probability':
	# 	if output_distribution_size != 2:
	# 		raise Exception('ERROR: Cannot interpret a target value as class probability for any other than two classes')
	#
	# 	bdy = [0.0,1.0]
	#
	# elif output_distribution_interpretation == 'regression':
	# 	interval = 1.0/output_distribution_size
	# 	bdy = list(frange(interval / 2, 1.0, interval))
	#
	# yi = np.sum(np.asarray(bdy)*dyi)
	# return yi
	return dyi[0]


def frange(start, stop=None, step=1.0, round_to=None):
	if stop == None:
		stop = start
		start = 0.0
	i = start
	while i < stop:
		if round_to is None:
			yield i
		else:
			yield round(i, round_to)
		i += step


class DictClass(OrderedDict):
	'''
	A subclass of OrderedDict that keeps its (ordered) items synced with its attributes
	'''

	def __init__(self, prefix='', *args):

		OrderedDict.__init__(self, args)
		self._prefix = prefix
		self._sync = True

	def __setattr__(self, name, value):
		if hasattr(self, '_sync') and self._sync:
			OrderedDict.__setitem__(self, name, value)

		OrderedDict.__setattr__(self, name, value)

	def __setitem__(self, key, val):
		if hasattr(self, '_sync') and self._sync:
			OrderedDict.__setattr__(self, key, val)
		OrderedDict.__setitem__(self, key, val)

	def __delattr__(self, name):
		if hasattr(self, '_sync') and self._sync:
			del self[name]
		del self.__dict__[name]

	def __str__(self, indent=0):
		rstring = ''
		for key in self.keys():
			if not str(key).startswith('_'):
				rstring += '\t' * indent
				if self._prefix:
					rstring += self._prefix + '_'
				rstring += str(key) + ": "
				val = self[key]
				if isinstance(val, DictClass):
					rstring += '\n' + val.__str__(indent=indent + 1)
				else:
					rstring += str(val) + '\n'
		return rstring


class SingleItemEvaluation(DictClass):
	def __init__(self, prefix=''):
		DictClass.__init__(self, prefix=prefix)

		self.true_y = None
		self.text = None
		self.predicted_rationale_evaluation = RationaleAndPredictionEvaluation(prefix='predicted_rationale')
		self.true_rationale_evaluation = TrueRationaleAndPredictionEvaluation(prefix='true_rationale')
		self.zero_rationale_evaluation = TrivialRationaleAndPredictionEvaluation(prefix='zero_rationale')
		self.one_rationale_evaluation = TrivialRationaleAndPredictionEvaluation(prefix='one_rationale')

	def __str__(self, indent=0, compare_predicted_to_true=True):
		rstring = ''
		for key in self.keys():
			if not str(key).startswith('_'):
				rstring += '\t' * indent
				if self._prefix:
					rstring += self._prefix + '_'
				rstring += str(key) + ": "
				val = self[key]
				if isinstance(val, DictClass):
					if compare_predicted_to_true and key == 'true_rationale_evaluation':
						rstring += '\n' + val.__str__(indent=indent + 1, compareto=self.predicted_rationale_evaluation)
					else:
						rstring += '\n' + val.__str__(indent=indent + 1)
				else:
					rstring += str(val) + '\n'
		return rstring


class DiscreteDatasetEvaluation(DictClass):
	def __init__(self, prefix=''):
		DictClass.__init__(self, prefix=prefix)

		self.y_accuracy = None
		self.y_precision = None
		self.y_recall = None
		self.y_f1 = None

		self.rationale_accuracy = None
		self.rationale_precision = None
		self.rationale_recall = None
		self.rationale_f1 = None


class DatasetEvaluation(DictClass):
	def __init__(self, prefix=''):
		DictClass.__init__(self, prefix=prefix)

		self.mean_y = None
		self.item_evaluations = []
		self.batch_evaluations = []
		self.mean_predicted_rationale_evaluation = RationaleAndPredictionEvaluation(prefix='mean_predicted_rationale',
																					mean=True)
		self.combined_predicted_rationale_evaluation = DiscreteDatasetEvaluation(prefix='combined_predicted')
		self.mean_true_rationale_evaluation = RationaleAndPredictionEvaluation(prefix='mean_true_rationale', mean=True)
		self.combined_baseline_evaluation = DiscreteDatasetEvaluation(prefix='combined_baseline')
		self.model_properties = ModelProperties()


class RationaleAndPredictionEvaluation(DictClass):
	def __init__(self, prefix='', mean=False):
		DictClass.__init__(self, prefix=prefix)

		self.predicted_y = None
		self.prediction_loss = None
		self.generator_loss = None
		self.encoder_loss = None
		self.inverse_encoder_loss = None
		self.inverse_predicted_y = None
		self.inverse_prediction_loss = None
		self.generator_inverse_loss = None
		self.generator_weighted_inverse_loss = None

		self.rationale = None
		self.probs = None
		self.rationalized_text = None
		self.accuracy = None
		self.precision = None
		self.recall = None
		self.f1 = None
		self.occlusion = None
		self.sparsity_loss = None
		self.weighted_sparsity_loss = None
		self.coherence_loss = None
		self.weighted_coherence_loss = None

		if mean:
			remove = [
				'rationale',
				'rationalized_text',
			]

			for field in remove:
				self.__delattr__(field)

	def __str__(self, indent=0, compareto=None):

		rstring = ''
		for key in self.keys():
			if not str(key).startswith('_'):
				rstring += '\t' * indent
				if self._prefix:
					rstring += self._prefix + '_'
				rstring += str(key) + ": "
				val = self[key]
				if isinstance(val, DictClass):
					rstring += '\n' + val.__str__(indent=indent + 1)
				else:
					if compareto and key in compareto:
						rstring += str(val) + ' ({} {})'.format(compareto[key], compareto._prefix) + '\n'
					else:
						rstring += str(val) + '\n'
		return rstring


class TrueRationaleAndPredictionEvaluation(RationaleAndPredictionEvaluation):
	def __init__(self, prefix='', mean=False):
		RationaleAndPredictionEvaluation.__init__(self, prefix=prefix, mean=mean)

		# Get rid of a few fields that don't really make sense for the true rationales
		remove = [
			'probs',
			'accuracy',
			'precision',
			'recall',
			'f1'
		]

		for field in remove:
			self.__delattr__(field)


class TrivialRationaleAndPredictionEvaluation(RationaleAndPredictionEvaluation):
	def __init__(self, prefix='', mean=False):
		RationaleAndPredictionEvaluation.__init__(self, prefix=prefix, mean=mean)

		keep = [
			'predicted_y',
			'generator_loss',
			'encoder_loss',
			'inverse_encoder_loss',
			'inverse_rationale_predicted_y'
		]

		for key in self.keys():
			if key not in keep:
				self.__delattr__(key)


class ModelProperties(DictClass):
	def __init__(self, prefix=''):
		DictClass.__init__(self, prefix=prefix)

		self.encoder_default_y = None
		self.inverse_encoder_default_y = None
		self.generator_l2_loss = None
		self.generator_l2_weight = None
		self.generator_weighted_l2_loss = None
		self.encoder_l2_loss = None
		self.encoder_l2_weight = None
		self.encoder_weighted_l2_loss = None
		self.inverse_encoder_l2_loss = None
		self.inverse_encoder_l2_weight = None
		self.inverse_encoder_weighted_l2_loss = None
		self.rationale_type = None
		self.prediction_loss_type = None
		self.inverse_encoder_confusion_method = None
		self.inverse_encoder_prediction_loss_type = None
		self.rationale_sparsity_loss_type = None
		self.rationale_sparsity_loss_weight = None
		self.rationale_coherence_loss_type = None
		self.rationale_coherence_loss_weight = None
		self.generator_inverse_encoder_loss_weight = None
		self.generator_inverse_encoder_loss_type = None


def unpad(v, padding_id, x=None):
	'''
	Take a numpy vector and remove all padding from it, returning a reduced vector
	:param v:
	:param padding_id:
	:return:
	'''
	try:
		if x is not None:
			assert (len(x) == len(v))
			return v[x != padding_id]
		else:
			return v[v != padding_id]
	except:
		pass


def dsum(d):
	return '\n'.join([str((k, v.shape, np.sum(v), np.mean(v))) for k, v in d.items()])


def invert(val, func, dtype):
	if func == 'sigmoid':
		return -np.log(1 / val - 1, dtype=dtype)
	elif func == 'tanh':
		return np.arctanh(val, dtype=dtype)
	else:
		raise Exception('Cannot invert unknown function {}'.format(func))


def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def calculate_mcnemar(true_y, py1, py2):
	acc1 = (true_y == py1).astype(int)
	acc2 = (true_y == py2).astype(int)

	a = np.sum((acc1 == 1) & (acc2 == 1))
	b = np.sum((acc1 == 1) & (acc2 == 0))
	c = np.sum((acc1 == 0) & (acc2 == 1))
	d = np.sum((acc1 == 0) & (acc2 == 0))

	if math.fabs(b - c) > 0:
		exact = False
	else:
		exact = True

	# if not exact:
	statistic = pow((b - c), 2) / float(b + c)
	p_val = stats.chi2.pdf(statistic, 1)
	# else:
	# 	statistic = None
	# 	p_val = 0
	# 	n = b+c
	# 	for i in range(b,n+1):
	# 		p_val += nchoosek(n,i)*pow(0.5,i)*pow(0.5,n-i)
	# 	p_val *= 2

	return statistic, p_val, (a, b, c, d)


def nchoosek(n, k):
	f = math.factorial
	return f(n) / f(k) / f(n - k)


def send_email(recipient, message, sender, subject=''):
	'''
	Send an email
	:param recipient:
	:param message:
	:param sender:
	:param subject:
	:return:
	'''
	try:
		s = smtplib.SMTP('localhost')
		msg = MIMEText(message, 'plain')
		msg['Subject'] = subject
		msg['From'] = sender
		msg['To'] = recipient
		s.sendmail(sender, [recipient], msg.as_string())
		s.quit()
	except Exception as ex:
		iprint('Could not send email.')
		traceback.print_exc()


def rationale_to_z_vector(rationale, dtype=np.float32, transpose=True):
	if transpose:
		return np.array([rationale]).astype(dtype).T
	else:
		return np.array(rationale).astype(dtype)


def add_rationales_to_df(df, rationales, seed=None, sample=100, dtype=np.float32, test_match=True):
	'''

	:param df: dataframe of a dataset. Must have a
	:param rationales: either a filename or a function. If a filename, it should be a csv with columns platform_comment_id and rationale, where the rationale is a json list of 1s and 0s. If a function it should map a text comment to a rationale which is a numpy vector
	:param sample: if this is not None, then sample this percentage of the original df and generate rationales for that sample only.
	:return:
	'''

	if type(rationales) == str:
		iprint('Loading rationales from file {}'.format(rationales))
		iinc()
		rationale_df = pd.read_csv(rationales)
		iprint('{} rationales found'.format(rationale_df.shape[0]))

		rationale_df['rationale'] = rationale_df['rationale'].apply(process_number_sequence_string)
		df = df.merge(rationale_df, left_on='id', right_on='platform_comment_id', how='left')

		num_rationales = df['rationale'].notnull().sum()
		iprint('{} rationales successfully matched to documents'.format(num_rationales))
		match_df = df[df['rationale'].notnull()]
		if match_df.shape[0] != rationale_df.shape[0]:
			iprint('ERROR: Only {} of {} rationales in rationale file could be matched to items in dataframe.'.format(
				match_df.shape[0], rationale_df.shape[0]))
		idec()

		if test_match:
			for row_num, row in match_df.iterrows():
				assert (len(row['text'].split()) == len(row['rationale']))

		return df
	# elif callable(rationales):
	# 	iprint('Generating synthetic rationales using function: {}'.format(rationales))
	# 	# if minimize_synthetic_rationales:
	# 	# 	iprint('Taking only the first token of each generated synthetic rationale')
	# 	# df['rationale'] = df['text'].sample(n=sample,random_state=seed).apply(rationales)
	#
	# 	df['rationale'] = rationales(df)
	# 	return df
	else:
		raise Exception("Type {} is not supported for rationales argument".format(type(rationales)))


nonnumeric = re.compile('[^0-9\.]+')


def process_number_sequence_string(s, func=float):
	if type(s) != str:
		return s
	else:
		v = np.array([func(n) for n in re.split(nonnumeric, s) if n != ''])
		return v


sequence_pattern = re.compile('\[ *([^0-9\.] ?)+\]')


def process_number_sequence_string_columns(df, columns, func=float):
	# iprint('Converting sequence string columns to arrays')
	iinc()
	for column in columns:
		# iprint(column)
		if column in df:
			# iprint('converting...',1)
			df[column] = df[column].apply(lambda v: process_number_sequence_string(v, func=func))
	idec()


def save_model_output(output_dict, dir):
	'''
	Model output is typically a dictionary of numpy arrays. This function saves each nparray as a file with filename corresponding to its key in the dictionary
	:return:
	'''

	iprint('Saving model results to {}'.format(dir))
	if not os.path.isdir(dir):
		os.makedirs(dir)
	iinc()
	for key, value in output_dict.items():
		iprint('{}...'.format(key))
		path = os.path.join(dir, '{}.npy'.format(key))
		try:
			with open(path, 'w') as f:
				np.save(f, value)
		except Exception as ex:
			iprint(ex.message)
	idec()


def load_model_output(dir):
	results = {}
	file_pattern = re.compile('(.+)\.npy')
	iprint('Loading model results from {}'.format(dir))
	iinc()
	for file in os.listdir(dir):
		m = re.match(file_pattern, file)
		if m:
			iprint('{}...'.format(file))
			key = m.groups()[0]
			values = np.load(os.path.join(dir, file))
			# if key in reshape_keys:
			# 	values = np.reshape(values,(-1,num_retrieval_gradients))
			results[key] = values
		else:
			iprint('File {} does not appear to be a saved numpy array'.format(file))
	idec()
	return results


validFilenameChars = "-_.() %s%s" % (string.ascii_letters, string.digits)


def slugify(filename, spaces_to_underscores=True):
	# cleanedFilename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore')
	cleanedFilename = ''.join(c for c in str(filename) if c in validFilenameChars)
	if spaces_to_underscores:
		cleanedFilename = cleanedFilename.replace(' ', '_')
	return cleanedFilename


def binarize(y, threshold=0.5):
	if type(y) == np.ndarray:
		if hasattr(y, 'astype'):
			return (y >= threshold).astype(np.float32)
		else:
			return float(y >= threshold)
	elif type(y) == list:
		return [1.0 if yi >= threshold else 0.0 for yi in y]
	else:
		return y


def main():
	test_split_params()


def hasrow(df, dic):
	'''
	Check if at least one row exists with column values equal to those in the given dictionary
	:param df:
	:param dic:
	:return:
	'''
	bool = True
	for k, v in dic.items():
		bool = bool & (df[k] == v)

	filtered_df = df[bool]
	return filtered_df.shape[0] > 0


def zipdict(dct):
	'''
	Assume dct is a dict where every value is an equal-sized list. Generate a list of dictionaries where each one has the corresponding element from each list as the value
	:param dct:
	:return:
	'''
	dlen = len(dct[list(dct.keys())[0]])

	return [{k: dct[k][i] for k in dct.keys()} for i in range(dlen)]


if __name__ == '__main__':
	main()


def subdir_paths(dir_path):
	try:
		filenames = os.listdir(dir_path)
		paths = [(filename, os.path.join(dir_path, filename)) for filename in filenames]
		subdir_paths = [(filename, filepath) for filename, filepath in paths if os.path.isdir(filepath)]
		return sorted(subdir_paths, key=lambda t: t[0])
	except:
		return []
