from cogdl.oag import oagbert
import torch
import os
import argparse
from tqdm import tqdm
import json
from collections import defaultdict

device = torch.device(0)

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='Art')
parser.add_argument('--metadata', default='None')
args = parser.parse_args()
dataset = args.dataset
mtype = args.metadata

if not os.path.exists(f'./Parabel/Sandbox/Data/{dataset}/'):
	os.makedirs(f'./Parabel/Sandbox/Data/{dataset}/')

tokenizer, model = oagbert("oagbert-v2")
# tokenizer, model = oagbert("oagbert-v2-sim")
model.to(device)
model.eval()

m2cnt = defaultdict(int)
label2idx = {}
train_cnt = test_cnt = 0
with open(f'../MAPLE/{dataset}/papers.json') as fin:
	for line in tqdm(fin):
		data = json.loads(line)
		if int(data['year']) <= 2015:
			train_cnt += 1
			for label in data['label']:
				if label not in label2idx:
					label2idx[label] = len(label2idx)
			ms = []
			if mtype == 'Venue':
				ms = [data['venue']]
			elif mtype == 'Author':
				ms = data['author']
			for m in ms:
				m2cnt[m] += 1
		else:
			test_cnt += 1
print(train_cnt, test_cnt, len(label2idx))

with open(f'./Parabel/Sandbox/Data/{dataset}/label2id.txt', 'w') as fout:
	for label in label2idx:
		fout.write(label+'\t'+str(label2idx[label])+'\n')

m2name = {}
if mtype == 'Venue':
	with open(f'../MAPLE/{dataset}/venues.txt') as fin:
		for line in fin:
			data = line.strip().split('\t')
			venue = data[0]
			name = data[2]
			m2name[venue] = name
elif mtype == 'Author':
	with open(f'../MAPLE/{dataset}/authors.txt') as fin:
		for line in fin:
			data = line.strip().split('\t')
			author = data[0]
			name = data[2]
			m2name[author] = name

with open(f'../MAPLE/{dataset}/papers.json') as fin, \
	 open(f'./Parabel/Sandbox/Data/{dataset}/trn_X_Xf.txt', 'w') as fou1, \
	 open(f'./Parabel/Sandbox/Data/{dataset}/trn_X_Y.txt', 'w') as fou2:
	fou1.write(str(train_cnt)+' 768\n')
	fou2.write(str(train_cnt)+' '+str(len(label2idx))+'\n')
	for line in tqdm(fin):
		data = json.loads(line)
		if int(data['year']) <= 2015:
			title = data['title_raw']
			abstract = data['abstract_raw']

			if mtype == 'Venue':
				if m2cnt[data['venue']] >= 5:
					venue = m2name[data['venue']]
				else:
					venue = ''
			else:
				venue = ''
			
			if mtype == 'Author':
				authors = [m2name[x] for x in data['author'] if m2cnt[x] >= 5]
			else:
				authors = []

			input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = model.build_inputs(
				title=title, abstract=abstract, venue=venue, authors=authors
			)
			sequence_output, pooled_output = model.bert.forward(
				input_ids=torch.LongTensor(input_ids).unsqueeze(0).to(device),
				token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).to(device),
				attention_mask=torch.LongTensor(input_masks).unsqueeze(0).to(device),
				output_all_encoded_layers=False,
				checkpoint_activations=False,
				position_ids=torch.LongTensor(position_ids).unsqueeze(0).to(device),
				position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0).to(device)
			)
			emb = sequence_output[0].mean(dim=0, keepdim=False).tolist()
			emb_str = [str(idx)+':'+str(round(x, 5)) for idx, x in enumerate(emb)]
			fou1.write(' '.join(emb_str)+'\n')

			label_str = []
			for label in data['label']:
				if label in label2idx:
					label_str.append(str(label2idx[label])+':1')		
			fou2.write(' '.join(label_str)+'\n')

			fou1.flush()
			fou2.flush()

with open(f'../MAPLE/{dataset}/papers.json') as fin, \
	 open(f'./Parabel/Sandbox/Data/{dataset}/tst_X_Xf.txt', 'w') as fou1, \
	 open(f'./Parabel/Sandbox/Data/{dataset}/tst_X_Y.txt', 'w') as fou2:
	fou1.write(str(test_cnt)+' 768\n')
	fou2.write(str(test_cnt)+' '+str(len(label2idx))+'\n')
	for line in tqdm(fin):
		data = json.loads(line)
		if int(data['year']) > 2015:
			title = data['title_raw']
			abstract = data['abstract_raw']

			if mtype == 'Venue':
				if m2cnt[data['venue']] >= 5:
					venue = m2name[data['venue']]
				else:
					venue = ''
			else:
				venue = ''
			
			if mtype == 'Author':
				authors = [m2name[x] for x in data['author'] if m2cnt[x] >= 5]
			else:
				authors = []

			input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = model.build_inputs(
				title=title, abstract=abstract, venue=venue, authors=authors
			)
			sequence_output, pooled_output = model.bert.forward(
				input_ids=torch.LongTensor(input_ids).unsqueeze(0).to(device),
				token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).to(device),
				attention_mask=torch.LongTensor(input_masks).unsqueeze(0).to(device),
				output_all_encoded_layers=False,
				checkpoint_activations=False,
				position_ids=torch.LongTensor(position_ids).unsqueeze(0).to(device),
				position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0).to(device)
			)
			emb = sequence_output[0].mean(dim=0, keepdim=False).tolist()
			emb_str = [str(idx)+':'+str(round(x, 5)) for idx, x in enumerate(emb)]
			fou1.write(' '.join(emb_str)+'\n')

			label_str = []
			for label in data['label']:
				if label in label2idx:
					label_str.append(str(label2idx[label])+':1')		
			fou2.write(' '.join(label_str)+'\n')

			fou1.flush()
			fou2.flush()
