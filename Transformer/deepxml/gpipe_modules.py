import torch
import torch.nn as nn
import torch.nn.functional as F

from deepxml.attentionxml import Embedding, LSTMEncoder
from deepxml.meshprobenet import MeSHProbes
from deepxml.cornet import CorNet
from deepxml.bertxml import BaseBertModel


class PlainC(nn.Module):
    def __init__(self, labels_num, context_size):
        super(PlainC, self).__init__()
        self.out_mesh_dstrbtn = nn.Linear(context_size, labels_num)
        nn.init.xavier_uniform_(self.out_mesh_dstrbtn.weight)

    def forward(self, context_vectors):
        output_dstrbtn = self.out_mesh_dstrbtn(context_vectors)  
        return output_dstrbtn

class MeSHProbeNet_encoder(nn.Module):
    def __init__(self, emb_size, hidden_size, n_layers, n_probes, labels_num, dropout, bottleneck_dim=None, 
                 vocab_size=None, emb_init=None, emb_trainable=True, padding_idx=0, emb_dropout=0.2, **kwargs):
        super(MeSHProbeNet_encoder, self).__init__()
        self.emb = Embedding(vocab_size, emb_size, emb_init, emb_trainable, padding_idx, emb_dropout)
        self.lstm = LSTMEncoder(emb_size, hidden_size, n_layers, dropout)
        self.meshprobes = MeSHProbes(hidden_size * 2, n_probes) # *2 because of bidirection
        self.pooler = nn.Linear(2 * hidden_size * n_probes, bottleneck_dim)
        nn.init.xavier_uniform_(self.pooler.weight)
            
    def forward(self, input_variables):
        emb_out, lengths, masks = self.emb(input_variables)
        birnn_outputs = self.lstm(emb_out, lengths)   # N, L, hidden_size * 2
        context_vectors = self.meshprobes(birnn_outputs, masks)
        context_vectors = F.elu(self.pooler(context_vectors))
        return context_vectors
    

class XMLCNN_encoder(nn.Module):
    def __init__(self, dropout, labels_num, dynamic_pool_length, bottleneck_dim, num_filters,
                 vocab_size=None, emb_size=None, emb_trainable=True, emb_init=None, padding_idx=0, **kwargs):
        super(XMLCNN_encoder, self).__init__()
        if emb_init is not None:
            if vocab_size is not None:
                assert vocab_size == emb_init.shape[0]
            if emb_size is not None:
                assert emb_size == emb_init.shape[1]
            vocab_size, emb_size = emb_init.shape            
        
        self.output_channel = num_filters
        self.num_bottleneck_hidden = bottleneck_dim
        self.dynamic_pool_length = dynamic_pool_length
            
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx, sparse=True,
                                _weight=torch.from_numpy(emb_init).float() if emb_init is not None else None)
        self.emb.weight.requires_grad = emb_trainable
        
        self.ks = 3 # There are three conv nets here
        ## Different filter sizes in xml_cnn than kim_cnn
        self.conv1 = nn.Conv2d(1, self.output_channel, (2, emb_size), padding=(1,0))
        self.conv2 = nn.Conv2d(1, self.output_channel, (4, emb_size), padding=(3,0))
        self.conv3 = nn.Conv2d(1, self.output_channel, (8, emb_size), padding=(7,0))
        self.pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length) #Adaptive pooling

        self.bottleneck = nn.Linear(self.ks * self.output_channel * self.dynamic_pool_length, self.num_bottleneck_hidden)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.bottleneck.weight)

    def forward(self, x):
        embe_out = self.emb(x) # (batch, sent_len, embed_dim)
        x = embe_out.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        x = [self.pool(i).squeeze(2) for i in x]

        # (batch, channel_output) * ks
        x = torch.cat(x, 1) # (batch, channel_output * ks)
        x = F.relu(self.bottleneck(x.view(-1, self.ks * self.output_channel * self.dynamic_pool_length)))
        context_vectors = self.dropout(x)
        return context_vectors


class gpipe_encoder(nn.Module):
    def __init__(self, model_name, **kwargs):
        super(gpipe_encoder, self).__init__()
        if 'MeSHProbeNet' in model_name:
            self.net = MeSHProbeNet_encoder(**kwargs)
        elif 'XMLCNN' in model_name:
            self.net = XMLCNN_encoder(**kwargs)
        elif 'BertXML' in model_name:
            self.net = BaseBertModel(**kwargs)
            
    def forward(self, input_variables):
        context_vectors = self.net(input_variables)    
        return context_vectors
    
class gpipe_decoder(nn.Module):
    def __init__(self, model_name, labels_num, bottleneck_dim, **kwargs):
        super(gpipe_decoder, self).__init__()
        if 'CorNet' not in model_name:
            self.net = PlainC(labels_num, bottleneck_dim)
        else:
            self.net = nn.Sequential(PlainC(labels_num, bottleneck_dim), CorNet(labels_num, **kwargs))
            
    def forward(self, context_vectors):
        logtis = self.net(context_vectors)    
        return logtis