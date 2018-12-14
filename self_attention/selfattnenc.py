import torch
from torch import optim
from functools import partial
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import os
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='MT Training')
parser.add_argument('--bhs', default = 512, type = int, help = 'Base Hidden Size')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
args = parser.parse_args()
base_embed_size = args.bhs
path_to_helper_files = os.path.join('..', 'py_files')
base_saved_models_dir = os.path.join('..', 'saved_models' )
sys.path.append(path_to_helper_files)
import global_variables
import dataset_helper
import nnet_models
import train_utilities
print('********************* Start Training *******************')
device = global_variables.device;
source_name = 'vi'
target_name = 'en'
MAX_LEN = 48
batchSize = 64
# base_embed_size = 512
source_embed_dim = base_embed_size
source_hidden_size = base_embed_size
source_rnn_layers = 2
source_rnn_type  = 'lstm'

target_embed_dim= base_embed_size
target_hidden_size = base_embed_size*2
target_rnn_layers = 2

attention = True
if source_name == 'vi' and target_name == 'en':
    target_train_path = '../Data/iwslt-vi-en/train.tok.en'
    source_train_path = '../Data/iwslt-vi-en/train.tok.vi'
    target_val_path = '../Data/iwslt-vi-en/dev.tok.en'
    source_val_path = '../Data/iwslt-vi-en/dev.tok.vi'
elif source_name == 'zh' and target_name == 'en':
    target_train_path = '../Data/iwslt-zh-en/train.tok.en'
    source_train_path = '../Data/iwslt-zh-en/train.tok.zh'
    target_val_path = '../Data/iwslt-zh-en/dev.tok.en'
    source_val_path = '../Data/iwslt-zh-en/dev.tok.zh'
else:
    sys.exit(source_name+'->'+target_name+' is invalid!')
saved_models_dir = os.path.join(base_saved_models_dir, source_name+'2'+target_name)
pth_save_folder_name = source_name+'2'+target_name+'_' + \
                'source_embed_dim='+str(source_embed_dim) + \
                'source_hidden_size='+str(source_hidden_size) + \
                'source_rnn_layers=' + str(source_rnn_layers) + \
                'source_rnn_type='+str(source_rnn_type)+ \
                'target_embed_dim='+str(target_embed_dim) + \
                'target_hidden_size='+str(target_hidden_size) + \
                'target_rnn_layers='+str(target_rnn_layers) + \
                'attention='+str(attention);
pth_saved_dir = os.path.join(saved_models_dir, pth_save_folder_name)
saved_language_model_dir = os.path.join(saved_models_dir, 'lang_obj')
dataset_dict = {'train': dataset_helper.LanguagePair(source_name = source_name, target_name=target_name,
                                                    source_path = source_train_path, target_path = target_train_path,
                                                    lang_obj_path = saved_language_model_dir ),
               'val': dataset_helper.LanguagePair(source_name = source_name, target_name=target_name,
                                                    source_path = source_val_path, target_path = target_val_path,
                                                    lang_obj_path = saved_language_model_dir, val = True ) }
dataloader_dict = {'train': DataLoader(dataset_dict['train'], batch_size = batchSize,
                                    collate_fn = partial(dataset_helper.vocab_collate_func, MAX_LEN=MAX_LEN),
                                shuffle = True, num_workers=0),
                  'val': DataLoader(dataset_dict['val'], batch_size = 1,
                                    collate_fn = dataset_helper.vocab_collate_func_val,
                                shuffle = True, num_workers=0)}
decoder = nnet_models.AttentionDecoderRNN(dataset_dict['train'].target_lang_obj.n_words,
                                            embed_dim = target_embed_dim,
                                            hidden_size = target_hidden_size,
                                            n_layers = target_rnn_layers,
                                            attention = attention).to(device)
criterion = nn.NLLLoss(ignore_index = global_variables.PAD_IDX)
# Vocab Size
V = dataset_dict['train'].source_lang_obj.n_words


def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class BiEncoder(nn.Module):
	"Core encoder is a stack of N layers"

	def __init__(self, layer, layer_n, N, src_embedding, src_enbedding_n):
		super(BiEncoder, self).__init__()
		self.encoder = Encoder(layer, N, src_embedding)
		self.encoder_n = Encoder(layer_n, N, src_embedding_n)
	def forward(self, x, x_len):
		"Pass the input (and mask) through each layer in turn."
		output, c, hidden = self.encoder(x, x_len)
		output_n, c_n, hidden_n = self.encoder_n(x, x_len)
		return torch.cat((output,output_n),2), torch.cat((c,c_n),2), torch.cat((hidden, hidden_n),2)
class Encoder(nn.Module):
	"Core encoder is a stack of N layers"

	def __init__(self, layer, N, src_embedding):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.src_embedding = src_embedding
		self.src_embedding_n = src_embedding_n
		self.norm = LayerNorm(layer.size)
	def forward(self, x, x_len):
		"Pass the input (and mask) through each layer in turn."
		mask = (x != global_variables.PAD_IDX).unsqueeze(-2)
		x = self.src_embedding(x)
		for layer in self.layers:
			x = layer(x, mask)
		x = self.norm(x)
		size1 = x.size(1)
		# cell first and last output
		c = torch.cat((x[:, 0, :].unsqueeze(0), x[:, -1, :].unsqueeze(0)), dim=0)
		# hidden sum over fisrt half and sum over the last half
		#         hidden1 = torch.sum(x[:,0:size1//2,:], dim = 1).unsqueeze(0)
		#         hidden2 = torch.sum(x[:,0:size1//2,:], dim = 1).unsqueeze(0)
		#         hidden = torch.cat((hidden1, hidden2), dim = 0)
		return x, torch.zeros(c.size()).to(device), torch.zeros(c.size()).to(device)


class LayerNorm(nn.Module):
	"Construct a layernorm module (See citation for details)."

	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.float().mean(-1, keepdim=True)
		std = x.float().std(-1, keepdim=True)
		return self.a_2 * (x.float() - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
	"""
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
	"Encoder is made up of self-attn and feed forward (defined below)"

	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size

	def forward(self, x, mask):
		"Follow Figure 1 (left) for connections."
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
	"Compute 'Scaled Dot Product Attention'"
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) \
			 / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key, value, mask=None):
		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)

		# 1) Do all the linear projections in batch from d_model => h x d_k
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.linears, (query, key, value))]

		# 2) Apply attention on all the projected vectors in batch.
		x, self.attn = attention(query, key, value, mask=mask,
								 dropout=self.dropout)

		# 3) "Concat" using a view and apply a final linear.
		x = x.transpose(1, 2).contiguous() \
			.view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)


class Embeddings(nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)
		self.d_model = d_model

	def forward(self, x):
		return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
	"Implement the PE function."

	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1).float()
		div_term = torch.exp(torch.arange(0, d_model, 2).float() *
							 -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)],
						 requires_grad=False)
		return self.dropout(x)


class NPositionalEncoding(nn.Module):
	"Implement the PE function."

	def __init__(self, d_model, dropout, max_len=5000):
		super(NPositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = (max_len - torch.arange(0, max_len)).unsqueeze(1).float()
		div_term = torch.exp(torch.arange(0, d_model, 2).float() *
							 -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)],
						 requires_grad=False)
		return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
	"Implements FFN equation."

	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.w_2(self.dropout(F.relu(self.w_1(x))))
import copy
c = copy.deepcopy
d_model=base_embed_size
d_ff=2048
h=8
dropout=0.1
N=6
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
attn = MultiHeadedAttention(h, d_model)
position = PositionalEncoding(d_model, dropout)
nposition = NPositionalEncoding(d_model, dropout)
embed = Embeddings(d_model, V)
src_embedding = nn.Sequential(embed, c(position))
src_embedding_n = nn.Sequential(embed, c(nposition))
encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
encoder_layer_n = EncoderLayer(d_model, c(attn), c(ff), dropout)
encoder = BiEncoder(encoder_layer, encoder_layer_n, N, src_embedding, src_embedding_n).to(device)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.25,nesterov=True, momentum = 0.99)
enc_scheduler = ReduceLROnPlateau(encoder_optimizer, min_lr=1e-4,  patience=0)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.25,nesterov=True, momentum = 0.99)
dec_scheduler = ReduceLROnPlateau(decoder_optimizer, min_lr=1e-4,  patience=0)
encoder, decoder, loss_hist, acc_hist = train_utilities.train_model(encoder_optimizer, decoder_optimizer,
                                            encoder, decoder, criterion,
                                            "attention", dataloader_dict, dataset_dict['train'].target_lang_obj,
                                            num_epochs = 100, rm = 0.95,
                                            enc_scheduler = enc_scheduler, dec_scheduler = dec_scheduler)
val_score = train_utilities.validation_function(encoder, decoder, dataloader_dict['val'],
                                                   dataset_dict['train'].target_lang_obj , attention)
save_models(encoder, decoder, pth_save_folder_name)