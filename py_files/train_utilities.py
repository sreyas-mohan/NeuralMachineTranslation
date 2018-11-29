import torch
import time
import random

import global_variables
from bleu_score import BLEU_SCORE


SOS_token = global_variables.SOS_token
EOS_token = global_variables.EOS_token
UNK_IDX = global_variables.UNK_IDX
PAD_IDX = global_variables.PAD_IDX

device = global_variables.device;

def convert_idx_2_sent(tensor, lang_obj):
	word_list = []
	for i in tensor:
		if i.item() not in set([ PAD_IDX, EOS_token, SOS_token]):
			word_list.append(lang_obj.index2word[i.item()])
	return (' ').join(word_list)

def validation(encoder, decoder, dataloader, loss_fun, lang_obj, max_len, m_type):
	encoder.train(False)
	decoder.train(False)
	pred_corpus = []
	true_corpus = []
	running_loss = 0
	running_total = 0
	bl = BLEU_SCORE()
	for data in dataloader:
		encoder_i = data[0].to(device)
		decoder_i = data[1].to(device)

		bs,sl = encoder_i.size()[:2]

		out, hidden = encode_decode(encoder,decoder,encoder_i,decoder_i,max_len, m_type, rand_num = 0)
		loss = loss_fun(out.float(), decoder_i.long())
		running_loss += loss.item() * bs
		running_total += bs
		pred = torch.max(out,dim = 1)[1]

		for t,p in zip(data[1],pred):
			t,p = convert_idx_2_sent(t, lang_obj), convert_idx_2_sent(p,lang_obj)
			true_corpus.append(t)
			pred_corpus.append(p)
	score = bl.corpus_bleu(pred_corpus,[true_corpus],lowercase=True)[0]
	return running_loss/running_total, score


def encode_decode(encoder, decoder, data_en, data_de, max_len, m_type, rand_num = 0.5):
	use_teacher_forcing = True if random.random() < rand_num else False
	bss = data_en.size(0)
	en_h = encoder.initHidden(bss)
	en_out,en_hid = encoder(data_en,en_h)
	
	decoder_hidden = en_hid
	decoder_input = torch.tensor([[SOS_token]]*bss).to(device)

	if use_teacher_forcing:
		d_out = []
		for i in range(max_len):
			if m_type=="attention":
				decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden,en_out)
			else:
				decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden)
			d_out.append(decoder_output.unsqueeze(-1))
			decoder_input = data_de[:,i].view(-1,1)
		d_hid = decoder_hidden
		d_out = torch.cat(d_out,dim=-1)
	else:
		d_out = []
		for i in range(max_len):
			if m_type == "attention":
				error('not implemented!')
				decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden,en_out)
			else:
				decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden)
			d_out.append(decoder_output.unsqueeze(-1))
			topv, topi = decoder_output.topk(1)
			decoder_input = topi.squeeze().detach().view(-1,1)
		d_hid = decoder_hidden
		d_out = torch.cat(d_out,dim=-1)
	return d_out, d_hid



def train_model(encoder_optimizer, decoder_optimizer, encoder, decoder, loss_fun,max_len, m_type, dataloader, target_lang_obj,
				num_epochs=60, val_every = 1, train_bleu_every = 10):
	best_score = 0
	best_bleu = 0
	loss_hist = {'train': [], 'val': []}
	bleu_hist = {'train': [], 'val': []}
	best_encoder_wts = None
	best_decoder_wts = None

	for epoch in range(num_epochs):

		start = time.time()
		total = 0
		top1_correct = 0
		running_loss = 0
		running_total = 0


		encoder.train(True)
		decoder.train(True)

		for data in dataloader['train']:
			encoder_optimizer.zero_grad()
			decoder_optimizer.zero_grad()

			encoder_i = data[0].to(device)
			decoder_i = data[1].to(device)
							
			out, hidden = encode_decode(encoder,decoder, encoder_i, decoder_i,max_len, m_type)

			loss = loss_fun(out.float(), decoder_i.long())

			N = decoder_i.size(0)
			running_loss += loss.item() * N
			
			total += N

			loss.backward()
			encoder_optimizer.step()
			decoder_optimizer.step()

		epoch_loss = running_loss / total
		loss_hist['train'].append(epoch_loss)
		print("epoch {} loss = {}, time = {}".format(epoch,  epoch_loss,
																	   time.time() - start))

		if epoch%val_every == 0:
			val_loss, val_bleu_score = validation(encoder,decoder, dataloader['val'], loss_fun, target_lang_obj ,max_len,m_type)
			loss_hist['val'].append(val_loss)
			bleu_hist['val'].append(val_bleu_score)
			print("val loss = ", val_loss)
			print("val BLEU = ", val_bleu_score)
			if val_bleu_score > best_bleu:
				best_bleu = val_bleu_score
				best_encoder_wts = encoder.state_dict()
				best_decoder_wts = decoder.state_dict()
		print('#'*50)

	encoder.load_state_dict(best_encoder_wts)
	decoder.load_state_dict(best_decoder_wts)
	print("Training completed. Best BLEU is {}".format(best_bleu))
	return encoder,decoder,loss_hist, bleu_hist


