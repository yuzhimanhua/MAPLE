import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deepxml.attentionxml import Embedding, LSTMEncoder
from deepxml.cornet import CorNet

    
class Probe(nn.Module):
    def __init__(self, dim, n_probes):
        super(Probe, self).__init__()
        self.self_attn = nn.Linear(dim, n_probes, bias=False)
        nn.init.xavier_uniform_(self.self_attn.weight)

    def forward(self, birnn_outputs, masks):
        attn = self.self_attn(birnn_outputs).transpose(1, 2).masked_fill(~masks, -np.inf)  # (batch, n_probes, in_len)
        attn = F.softmax(attn, -1)
        # (batch, n_probes, in_len) * (batch, in_len, dim) -> (batch, n_probes, dim)
        context_vectors = torch.bmm(attn, birnn_outputs)
        return context_vectors
    
    
class MeSHProbes(nn.Module):    
    def __init__(self, hidden_size, n_probes):
        super(MeSHProbes, self).__init__()
        self.probes = Probe(hidden_size, n_probes)

    def forward(self, birnn_outputs, masks):
        masks = torch.unsqueeze(masks, 1)  # (batch, 1, in_len)
        batch_size = birnn_outputs.size(0)
        context_vectors = self.probes(birnn_outputs, masks)
        context_vectors = context_vectors.view(batch_size, -1) # (batch, n_probes * dim)
        return context_vectors
    

class PlainC(nn.Module):
    def __init__(self, labels_num, hidden_size, n_probes):
        super(PlainC, self).__init__()
        self.out_mesh_dstrbtn = nn.Linear(hidden_size * n_probes, labels_num)
        nn.init.xavier_uniform_(self.out_mesh_dstrbtn.weight)

    def forward(self, context_vectors):
        output_dstrbtn = self.out_mesh_dstrbtn(context_vectors)  
        return output_dstrbtn


class MeSHProbeNet(nn.Module):
    def __init__(self, emb_size, hidden_size, n_layers, n_probes, labels_num, dropout, 
                 vocab_size=None, emb_init=None, emb_trainable=True, padding_idx=0, emb_dropout=0.2, **kwargs):
        super(MeSHProbeNet, self).__init__()
        self.emb = Embedding(vocab_size, emb_size, emb_init, emb_trainable, padding_idx, emb_dropout)
        self.lstm = LSTMEncoder(emb_size, hidden_size, n_layers, dropout)
        self.meshprobes = MeSHProbes(hidden_size * 2, n_probes) # *2 because of bidirection
        self.plaincls = PlainC(labels_num, hidden_size * 2, n_probes)
            
    def forward(self, input_variables):
        emb_out, lengths, masks = self.emb(input_variables)
        birnn_outputs = self.lstm(emb_out, lengths)   # N, L, hidden_size * 2
        context_vectors = self.meshprobes(birnn_outputs, masks)   
        logits = self.plaincls(context_vectors)
        return logits
    
    
class CorNetMeSHProbeNet(nn.Module):
    def __init__(self, emb_size, hidden_size, n_layers, n_probes, labels_num, dropout, **kwargs):
        super(CorNetMeSHProbeNet, self).__init__()
        self.meshprobenet = MeSHProbeNet(emb_size, hidden_size, n_layers, n_probes, labels_num, dropout, **kwargs)  
        self.cornet = CorNet(labels_num, **kwargs)
            
    def forward(self, input_variables):
        raw_logits = self.meshprobenet(input_variables)
        cor_logits = self.cornet(raw_logits)        
        return cor_logits
