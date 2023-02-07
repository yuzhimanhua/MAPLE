import json
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='Biology_MeSH')
args = parser.parse_args()
dataset = args.dataset

if not os.path.exists(f'./Parabel/Sandbox/Data/{dataset}/'):
	os.makedirs(f'./Parabel/Sandbox/Data/{dataset}/')

label2names = {}
label2tokens = {}
with open(f'../MAPLE/{dataset}/labels_mesh.txt') as fin:
	for line in tqdm(fin):
		data = line.strip().split('\t')
		label = data[0]
		names = data[1:]
		tokens = [set(x.split()) for x in names]
		label2names[label] = names
		label2tokens[label] = tokens

with open(f'../MAPLE/{dataset}/papers.json') as fin, \
	 open(f'./Parabel/Sandbox/Data/{dataset}/matched_labels.txt', 'w') as fout:
	fout.write(str(len(label2names))+'\n')
	for line in tqdm(fin):
		data = json.loads(line)
		year = int(data['year'])
		if year <= 2015:
			continue
		text = (data['title'] + ' ' + data['abstract']).strip()
		tokens = set(text.split())
		candidates = []
		for label in label2names:
			for name, tset in zip(label2names[label], label2tokens[label]):
				matched = 1
				for t in tset:
					if t not in tokens:
						matched = 0
						break
				if matched == 1 and len(tset) > 1:
					if name not in text:
						matched = 0
				if matched == 1:
					candidates.append(label)
					break
		fout.write('\t'.join(candidates)+'\n')