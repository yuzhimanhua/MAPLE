import json
import os
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='Art')
parser.add_argument('--metadata', default='None')
args = parser.parse_args()
dataset = args.dataset
mtype = args.metadata

def get_input(data, m2cnt, mtype):
	text = data['title'] + ' ' + data['abstract']

	if mtype == 'None':
		return text.strip()

	if mtype == 'Venue':
		ms = ['VENUE_'+data['venue']]
	elif mtype == 'Author':
		ms = ['AUTHOR_'+x for x in data['author']]
	elif mtype == 'Reference':
		ms = ['REFERENCE_'+x for x in data['reference']]
	
	meta_filtered = [m for m in ms if m2cnt[m] >= 5] + ['[SEP]']
	meta_seq = ' '.join(meta_filtered)
	final_seq = meta_seq + ' ' + text
	return final_seq.strip()

if not os.path.exists(f'data/{dataset}/'):
	os.makedirs(f'data/{dataset}/')

m2cnt = defaultdict(int)
with open(f'../MAPLE/{dataset}/papers.json') as fin:
	for line in fin:
		data = json.loads(line)
		if int(data['year']) <= 2015:
			ms = ['VENUE_'+data['venue']]
			ms += ['AUTHOR_'+x for x in data['author']]
			ms += ['REFERENCE_'+x for x in data['reference']]
			for m in ms:
				m2cnt[m] += 1

train_and_dev_cnt = 0
with open(f'../MAPLE/{dataset}/papers.json') as fin, \
	 open(f'data/{dataset}/train_texts.txt', 'w') as fou1, \
	 open(f'data/{dataset}/train_labels.txt', 'w') as fou2:
	for line in fin:
		data = json.loads(line)
		if int(data['year']) <= 2015:
			text = get_input(data, m2cnt, mtype)
			label = ' '.join(data['label'])
			fou1.write(text+'\n')
			fou2.write(label+'\n')
			train_and_dev_cnt += 1

print(train_and_dev_cnt)

with open(f'../MAPLE/{dataset}/papers.json') as fin, \
	 open(f'data/{dataset}/test_texts.txt', 'w') as fou1, \
	 open(f'data/{dataset}/test_labels.txt', 'w') as fou2:
	for line in fin:
		data = json.loads(line)
		if int(data['year']) > 2015:
			text = get_input(data, m2cnt, mtype)
			label = ' '.join(data['label'])
			fou1.write(text+'\n')
			fou2.write(label+'\n')