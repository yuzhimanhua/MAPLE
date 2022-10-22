import json
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='Art')
args = parser.parse_args()
dataset = args.dataset

if not os.path.exists(f'./Parabel/Sandbox/Data/{dataset}/'):
	os.makedirs(f'./Parabel/Sandbox/Data/{dataset}/')

label2name = {}
label2token = {}
with open(f'../MAPLE/{dataset}/labels.txt') as fin:
	for line in tqdm(fin):
		data = line.strip().split('\t')
		label = data[0]
		name = data[1]
		label2name[label] = name
		label2token[label] = set(name.split())

with open(f'../MAPLE/{dataset}/papers.json') as fin, \
	 open(f'./Parabel/Sandbox/Data/{dataset}/matched_labels.txt', 'w') as fout:
	fout.write(str(len(label2name))+'\n')
	for line in tqdm(fin):
		data = json.loads(line)
		year = int(data['year'])
		if year <= 2015:
			continue
		text = (data['title'] + ' ' + data['abstract']).strip()
		tokens = set(text.split())
		candidates = []
		for label in label2name:
			matched = 1
			for token in label2token[label]:
				if token not in tokens:
					matched = 0
					break
			if matched == 1 and len(label2token[label]) > 1:
				if label2name[label] not in text:
					matched = 0
			if matched == 1:
				candidates.append(label)
		fout.write('\t'.join(candidates)+'\n')