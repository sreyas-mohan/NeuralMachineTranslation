import torch
import time
import random
import errno
import sys
import copy
import numpy as np

import global_variables
from bleu_score import BLEU_SCORE


SOS_token = global_variables.SOS_token
EOS_token = global_variables.EOS_token
UNK_IDX = global_variables.UNK_IDX
PAD_IDX = global_variables.PAD_IDX

device = global_variables.device;

def make_sure_path_exists(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise
			
			
def convert_id_list_2_sent(list_idx, lang_obj):
	word_list = []
	if type(list_idx) == list:
		for i in list_idx:
			if i not in set([EOS_token]):
				word_list.append(lang_obj.index2word[i])
	else:
		for i in list_idx:
			if i.item() not in set([EOS_token,SOS_token,PAD_IDX]):
				word_list.append(lang_obj.index2word[i.item()])
	return (' ').join(word_list)


def validation_function(encoder, decoder, val_dataloader, lang_en,  verbose = False, keep_unk = False, return_attention = False):
	encoder.eval()
	decoder.eval()
	pred_corpus = []
	true_corpus = []
	running_loss = 0
	running_total = 0
	bl = BLEU_SCORE()
	attention_scores_for_all_val = []
	for data in val_dataloader:
		encoder_i = data[0].to(device)
		src_len = data[2].to(device)
		bs,sl = encoder_i.size()[:2]
		en_out,en_hid,en_c = encoder(encoder_i,src_len)
		max_src_len_batch = max(src_len).item()
		prev_hiddens = en_hid
		prev_cs = en_c
		decoder_input = torch.tensor([[SOS_token]]*bs).to(device)
		prev_output = torch.zeros((bs, en_out.size(-1))).to(device)
		d_out = []
		
		attention_scores = []
		
		for i in range(sl*2):
			out_vocab, prev_output,prev_hiddens, prev_cs, attention_score = decoder(decoder_input,prev_output, \
																					prev_hiddens,prev_cs, en_out,\
																					src_len)
			topv, topi = out_vocab.topk(1)
#             decoder_input = topi.squeeze().detach().view(-1,1)
			d_out.append(topi.item())
			decoder_input = topi.squeeze().detach().view(-1,1)
		
			if decoder.att_layer is not None and return_attention:
				attention_scores.append(attention_score.unsqueeze(-1))
			if topi.item() == EOS_token:
				break
		
		if keep_unk:
			true_sent = convert_id_list_2_sent(data[1][0],lang_en)
			true_corpus.append(true_sent)
		else:
			true_corpus.append(data[-1])

		pred_sent = convert_id_list_2_sent(d_out,lang_en)
		pred_corpus.append(pred_sent)
		
		if decoder.att_layer is not None and return_attention:
			attention_scores = torch.cat(attention_scores, dim = -1)
			attention_scores_for_all_val.append(attention_scores)
			
		if verbose:
			print("True Sentence:",data[-1])
			print("Pred Sentence:", pred_sent)
			print('-*'*50)
	score = bl.corpus_bleu(pred_corpus,[true_corpus],lowercase=True)[0]

	if decoder.att_layer is not None and return_attention:
		return score, true_corpus, pred_corpus, attention_scores_for_all_val
	
	return score


def validation_beam_search(encoder, decoder, val_dataloader,lang_en,beam_size, verbose = False,
						   keep_unk = False, return_attention = False):
	encoder.eval()
	decoder.eval()
	encoder = encoder.to(device)
	decoder = decoder.to(device)
	pred_corpus = []
	true_corpus = []
	running_loss = 0
	running_total = 0
	bl = BLEU_SCORE()
	
	attention_scores_for_all_val = []
	
	j = 0
	for data in val_dataloader:

		encoder_i = data[0].to(device)
		src_len = data[2].to(device)
		bs,sl = encoder_i.size()[:2]
		en_out,en_hid,en_c = encoder(encoder_i,src_len)
		max_src_len_batch = max(src_len).item()
		prev_hiddens = en_hid
		prev_cs = en_c
		decoder_input = torch.tensor([[SOS_token]]*bs).to(device)
		prev_output = torch.zeros((bs, en_out.size(-1))).to(device)
		list_decoder_input = [None]*beam_size
		beam_stop_flags = [False]*beam_size
		beam_score = torch.zeros((bs,beam_size)).to(device)
		list_d_outs = [[] for _ in range(beam_size)]
		select_beam_size = beam_size
		
		attention_scores = [[] for _ in range(beam_size)]
		
		for i in range(sl+20):
			if i == 0:
				out_vocab, prev_output,prev_hiddens, prev_cs, attention_score = decoder(decoder_input,prev_output, \
																					prev_hiddens,prev_cs, en_out,\
																					src_len)
				bss, vocab_size = out_vocab.size()
				topv, topi = out_vocab.topk(beam_size)
				list_prev_output = [prev_output]*beam_size
				list_prev_hiddens = [prev_hiddens]*beam_size
				list_prev_cs = [prev_cs]*beam_size
				for b in range(beam_size):
					beam_score[0][b] = topv[0][b].item()
					list_decoder_input[b] = topi[0][b].squeeze().detach().view(-1,1)
					list_d_outs[b].append(topi[0][b].item())
					
					if decoder.att_layer is not None and return_attention:
						attention_scores[b].append(attention_score.unsqueeze(-1))
						
					if topi[0][b].item() == EOS_token:
						beam_stop_flags[b] = True
			else:
				beam_out_vocab = [None]*beam_size
				temp_out = [None]*beam_size
				temp_hid = [None]*beam_size
				temp_c = [None]*beam_size
				prev_d_outs = copy.deepcopy(list_d_outs)
				for b in range(beam_size):
					if not beam_stop_flags[b]:
						beam_out_vocab[b], temp_out[b], temp_hid[b], temp_c[b], attention_score =\
							decoder(list_decoder_input[b],list_prev_output[b],list_prev_hiddens[b],list_prev_cs[b],\
									en_out,src_len)
						beam_out_vocab[b] = beam_out_vocab[b] + beam_score[0][b]
					if beam_stop_flags[b]:
						beam_out_vocab[b] = torch.zeros(bss,vocab_size).fill_(float('-inf')).to(device)
				beam_out_vocab = torch.cat(beam_out_vocab,dim = 1)
				
				topv, topi = beam_out_vocab.topk(beam_size)
				id_for_hid = topi//vocab_size
				topi_idx = topi%vocab_size
				for b in range(beam_size):
					if not beam_stop_flags[b]:
						beam_score[0][b] = topv[0][b].item()
						list_decoder_input[b] = topi_idx[0][b].squeeze().detach().view(-1,1)
						list_d_outs[b] = copy.deepcopy(prev_d_outs[id_for_hid[0][b]])
						list_d_outs[b].append(topi_idx[0][b].item())
						if topi_idx[0][b].item() == EOS_token:
							beam_stop_flags[b] = True
						else:
							list_prev_output[b] = temp_out[id_for_hid[0][b]]
							list_prev_hiddens[b] = temp_hid[id_for_hid[0][b]]
							list_prev_cs[b] = temp_c[id_for_hid[0][b]]
				if all(beam_stop_flags):
					break
		id_max_score = np.argmax(beam_score)
		d_out = list_d_outs[id_max_score]
		
		if keep_unk:
			true_sent = convert_id_list_2_sent(data[1][0],lang_en)
			true_corpus.append(true_sent)
		else:
			true_corpus.append(data[-1])

		pred_sent = convert_id_list_2_sent(d_out,lang_en)
		pred_corpus.append(pred_sent)
		
		if decoder.att_layer is not None and return_attention:
			attention_scores = torch.cat(attention_scores, dim = -1)
			attention_scores_for_all_val.append(attention_scores)
			
		if verbose:
			print("True Sentence:",data[-1])
			print("Pred Sentence:", pred_sent)
			print('-*'*50)
	score = bl.corpus_bleu(pred_corpus,[true_corpus],lowercase=True)[0]
	
	if decoder.att_layer is not None and return_attention:
		return score, true_corpus, pred_corpus, attention_scores_for_all_val

	return score


def encode_decode(encoder, decoder, data_en, data_de,
					src_len,tar_len,rand_num = 0.95, val = False):
	
	if not val:
		use_teacher_forcing = True if random.random() < rand_num else False

		bss = data_en.size(0)
		en_out, en_hid, en_c = encoder(data_en, src_len)
		max_src_len_batch = max(src_len).item()
		max_tar_len_batch = max(tar_len).item()
		
# 		print(max_src_len_batch, max_tar_len_batch)
		prev_hiddens = en_hid
		prev_cs = en_c
		decoder_input = torch.tensor([[SOS_token]]*bss).to(device)
		prev_output = torch.zeros((bss, en_out.size(-1))).to(device)

		if use_teacher_forcing:
			d_out = []
			for i in range(max_tar_len_batch):
				out_vocab, prev_output,prev_hiddens, prev_cs, attention_score = decoder(decoder_input,prev_output, \
																						prev_hiddens,prev_cs, en_out,\
																						src_len)
				d_out.append(out_vocab.unsqueeze(-1))
				decoder_input = data_de[:,i].view(-1,1)
			d_out = torch.cat(d_out,dim=-1)

		else:
			d_out = []
			for i in range(max_tar_len_batch):
				out_vocab, prev_output,prev_hiddens, prev_cs, attention_score = decoder(decoder_input,prev_output, \
																						prev_hiddens,prev_cs, en_out,\
																						src_len)
				d_out.append(out_vocab.unsqueeze(-1))
				topv, topi = out_vocab.topk(1)
				decoder_input = topi.squeeze().detach().view(-1,1)

			d_out = torch.cat(d_out,dim=-1)

		return d_out


	else:


		encoder.eval()
		decoder.eval()
		bss = data_en.size(0)
		en_out,en_hid,en_c = encoder(data_en, src_len)
		max_src_len_batch = max(src_len).item()
		max_tar_len_batch = max(tar_len).item()
		prev_hiddens = en_hid
		prev_cs = en_c
		decoder_input = torch.tensor([[SOS_token]]*bss).to(device)
		prev_output = torch.zeros((bss, en_out.size(-1))).to(device)
		d_out = []
		for i in range(max_tar_len_batch):
			out_vocab, prev_output,prev_hiddens, prev_cs, attention_score = decoder(decoder_input,prev_output, \
																					prev_hiddens,prev_cs, en_out,\
																					src_len)
			d_out.append(out_vocab.unsqueeze(-1))
			topv, topi = out_vocab.topk(1)
			decoder_input = topi.squeeze().detach().view(-1,1)
		d_out = torch.cat(d_out,dim=-1)
		return d_out




def train_model(encoder_optimizer, decoder_optimizer, encoder, decoder, loss_fun, attention,  dataloader, en_lang,\
				num_epochs=60, val_every = 1, train_bleu_every = 10, clip = 0.1, rm = 0.8, enc_scheduler = None,\
			   dec_scheduler = None):


	best_bleu = -1
	loss_hist = {'train': [], 'val': []}
	bleu_hist = {'train': [], 'val': []}
	best_encoder_wts = encoder.state_dict()
	best_decoder_wts = decoder.state_dict()

	if attention:
		assert( decoder.att_layer is not None)
	else:
		assert( decoder.att_layer is None)

	phases = ['train','val']

	for epoch in range(num_epochs):

		for ex, phase in enumerate(phases):
			start = time.time()
			total = 0
			top1_correct = 0
			running_loss = 0
			running_total = 0

			if phase == 'train':
				encoder.train()
				decoder.train()
			else:
				encoder.eval()
				decoder.eval()

			for data in dataloader[phase]:
				encoder_optimizer.zero_grad()
				decoder_optimizer.zero_grad()

				encoder_i = data[0].to(device)
				decoder_i = data[1].to(device)
				src_len = data[2].to(device)
				tar_len = data[3].to(device)

				if phase == 'val':                
					out = encode_decode(encoder, decoder, encoder_i,decoder_i,src_len,tar_len, rand_num=rm, val = True )
				else:
					out = encode_decode(encoder, decoder, encoder_i,decoder_i,src_len,tar_len, rand_num=rm, val = False )
					
				N = decoder_i.size(0)

# 				print(out.shape)
# 				print(decoder_i.shape)
				loss = loss_fun(out.float(), decoder_i.long() )
				running_loss += loss.item() * N
				
				total += N
				if phase == 'train':
					loss.backward()
					torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
					torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
					encoder_optimizer.step()
					decoder_optimizer.step()
					
			epoch_loss = running_loss / total 
			loss_hist[phase].append(epoch_loss)
			print("epoch {} {} loss = {}, time = {}".format(epoch, phase, epoch_loss,
																		   time.time() - start))
			sys.stdout.flush()
		if (enc_scheduler is not None) and (dec_scheduler is not None):
			enc_scheduler.step(epoch_loss)
			dec_scheduler.step(epoch_loss)

		if epoch%val_every == 0:
			val_bleu_score = validation_function(encoder,decoder, dataloader['val'], en_lang)
			bleu_hist['val'].append(val_bleu_score)
			print("validation BLEU = ", val_bleu_score)
			if val_bleu_score > best_bleu:
				best_bleu = val_bleu_score
				best_encoder_wts = encoder.state_dict()
				best_decoder_wts = decoder.state_dict()

		print('='*50)

	encoder.load_state_dict(best_encoder_wts)
	decoder.load_state_dict(best_decoder_wts)
	print("Training completed. Best BLEU is {}".format(best_bleu))

	return encoder,decoder,loss_hist,bleu_hist


