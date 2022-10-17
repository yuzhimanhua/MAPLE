import json
import os
from collections import defaultdict
import argparse
import math

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='Art')
parser.add_argument('--metadata', default='None')
args = parser.parse_args()
dataset = args.dataset
mtype = args.metadata

if not os.path.exists(f'./Sandbox/Data/{dataset}/'):
	os.makedirs(f'./Sandbox/Data/{dataset}/')

word2cnt = defaultdict(int)
label2idx = {}
train_cnt = test_cnt = 0
with open(f'../MAPLE/{dataset}/papers.json') as fin:
	for line in fin:
		data = json.loads(line)
		if int(data['year']) <= 2015:
			words = data['title'].split() + data['abstract'].split()
			words = list(set(words))
			if mtype == 'Venue':
				words += ['VENUE_'+data['venue']]
			elif mtype == 'Author':
				words += ['AUTHOR_'+x for x in data['author']]
			elif mtype == 'Reference':
				words += ['REFERENCE_'+x for x in data['reference']]
			for word in words:
				word2cnt[word] += 1
			for label in data['label']:
				if label not in label2idx:
					label2idx[label] = len(label2idx)
			train_cnt += 1
		else:
			test_cnt += 1

word2idx = {}
word2idf = {}
for word in word2cnt:
	if word2cnt[word] >= 5:
		word2idx[word] = len(word2idx)
		word2idf[word] = math.log(train_cnt/word2cnt[word])

print(train_cnt, test_cnt, len(word2idx), len(label2idx))

with open(f'../MAPLE/{dataset}/papers.json') as fin, \
	 open(f'./Sandbox/Data/{dataset}/trn_X_Xf.txt', 'w') as fou1, \
	 open(f'./Sandbox/Data/{dataset}/trn_X_Y.txt', 'w') as fou2:
	fou1.write(str(train_cnt)+' '+str(len(word2idx))+'\n')
	fou2.write(str(train_cnt)+' '+str(len(label2idx))+'\n')
	for line in fin:
		data = json.loads(line)
		if int(data['year']) <= 2015:
			words = data['title'].split() + data['abstract'].split()
			if mtype == 'Venue':
				words += ['VENUE_'+data['venue']]
			elif mtype == 'Author':
				words += ['AUTHOR_'+x for x in data['author']]
			elif mtype == 'Reference':
				words += ['REFERENCE_'+x for x in data['reference']]
			bow = defaultdict(float)
			for word in words:
				if word in word2idx:
					bow[word2idx[word]] += word2idf[word]
			bow_str = []
			for word in bow:
				bow_str.append(str(word)+':'+str(bow[word]))
			fou1.write(' '.join(bow_str)+'\n')

			label_str = []
			for label in data['label']:
				if label in label2idx:
					label_str.append(str(label2idx[label])+':1')		
			fou2.write(' '.join(label_str)+'\n')

with open(f'../MAPLE/{dataset}/papers.json') as fin, \
	 open(f'./Sandbox/Data/{dataset}/tst_X_Xf.txt', 'w') as fou1, \
	 open(f'./Sandbox/Data/{dataset}/tst_X_Y.txt', 'w') as fou2:
	fou1.write(str(test_cnt)+' '+str(len(word2idx))+'\n')
	fou2.write(str(test_cnt)+' '+str(len(label2idx))+'\n')
	for line in fin:
		data = json.loads(line)
		if int(data['year']) > 2015:
			words = data['title'].split() + data['abstract'].split()
			if mtype == 'Venue':
				words += ['VENUE_'+data['venue']]
			elif mtype == 'Author':
				words += ['AUTHOR_'+x for x in data['author']]
			elif mtype == 'Reference':
				words += ['REFERENCE_'+x for x in data['reference']]
			bow = defaultdict(float)
			for word in words:
				if word in word2idx:
					bow[word2idx[word]] += word2idf[word]
			bow_str = []
			for word in bow:
				bow_str.append(str(word)+':'+str(bow[word]))
			fou1.write(' '.join(bow_str)+'\n')

			label_str = []
			for label in data['label']:
				if label in label2idx:
					label_str.append(str(label2idx[label])+':1')		
			fou2.write(' '.join(label_str)+'\n')