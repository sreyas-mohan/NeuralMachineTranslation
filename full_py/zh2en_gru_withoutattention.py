import torch
from torch import optim
from functools import partial
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import os
import torch.nn as nn
import numpy as np



path_to_helper_files = os.path.join('..', 'py_files')
base_saved_models_dir = os.path.join('..', 'saved_models' )



sys.path.append(path_to_helper_files)


import global_variables
import dataset_helper
import nnet_models
import train_utilities

device = global_variables.device;





source_name = 'zh'
target_name = 'en'




MAX_LEN = 48
batchSize = 128


source_rnn_type  = 'gru'



attention = False



embed_dim_array = [512, 256]
rnn_layers_array = [2, 1]

for embed_dim in embed_dim_array:
	for rnn_layers in rnn_layers_array:

		source_embed_dim = embed_dim
		source_hidden_size = embed_dim

		target_embed_dim= embed_dim
		target_hidden_size = 2*embed_dim
		source_rnn_layers = rnn_layers
		target_rnn_layers = rnn_layers


		if source_name == 'vi' and target_name == 'en':
				target_train_path = '../Data/iwslt-vi-en/train.tok.en'
				source_train_path = '../Data/iwslt-vi-en/train.tok.vi'

				target_val_path = '../Data/iwslt-vi-en/dev.tok.en'
				source_val_path = '../Data/iwslt-vi-en/dev.tok.vi'

				target_test_path = '../Data/iwslt-vi-en/test.tok.en'
				source_test_path = '../Data/iwslt-vi-en/test.tok.vi'

		elif source_name == 'zh' and target_name == 'en':
				target_train_path = '../Data/iwslt-zh-en/train.tok.en'
				source_train_path = '../Data/iwslt-zh-en/train.tok.zh'

				target_val_path = '../Data/iwslt-zh-en/dev.tok.en'
				source_val_path = '../Data/iwslt-zh-en/dev.tok.zh'

				target_test_path = '../Data/iwslt-zh-en/test.tok.en'
				source_test_path = '../Data/iwslt-zh-en/test.tok.zh'
		else:
				sys.exit(source_name+'->'+target_name+' is invalid!')




		saved_models_dir = os.path.join(base_saved_models_dir, source_name+'2'+target_name)




		pth_save_folder_name = source_name+'2'+target_name+'_' + \
														'source_embed_dim='+str(source_embed_dim) +  \
														'-source_hidden_size='+str(source_hidden_size) +  \
														'-source_rnn_layers=' + str(source_rnn_layers) + \
														'-source_rnn_type='+str(source_rnn_type)+ \
														'-target_embed_dim='+str(target_embed_dim) + \
														'-target_hidden_size='+str(target_hidden_size) + \
														'-target_rnn_layers='+str(target_rnn_layers) + \
														'-attention='+str(attention);
		pth_saved_dir = os.path.join(saved_models_dir, pth_save_folder_name)



		print(pth_save_folder_name)
		sys.stdout.flush()

		saved_language_model_dir = os.path.join(saved_models_dir, 'lang_obj')



		dataset_dict = {'train': dataset_helper.LanguagePair(source_name = source_name, target_name=target_name, 
																												source_path = source_train_path, target_path = target_train_path, 
																												lang_obj_path = saved_language_model_dir), 

					 	'val': dataset_helper.LanguagePair(source_name = source_name, target_name=target_name, 
																								source_path = source_val_path, target_path = target_val_path, 
																								lang_obj_path = saved_language_model_dir, val = True), 

						'test': dataset_helper.LanguagePair(source_name = source_name, target_name=target_name, 
																								source_path = source_test_path, target_path = target_test_path, 
																								lang_obj_path = saved_language_model_dir, val = True)}																	  






		dataloader_dict = {'train': DataLoader(dataset_dict['train'], batch_size = batchSize, 
																				collate_fn = partial(dataset_helper.vocab_collate_func, MAX_LEN=MAX_LEN),
																		shuffle = True, num_workers=0), 
											'val': DataLoader(dataset_dict['val'], batch_size = 1, 
																				collate_fn = dataset_helper.vocab_collate_func_val,
																		shuffle = True, num_workers=0), 
											'test': DataLoader(dataset_dict['test'], batch_size = 1, 
																				collate_fn = dataset_helper.vocab_collate_func_val,
																		shuffle = True, num_workers=0)}





		encoder = nnet_models.EncoderRNN(dataset_dict['train'].source_lang_obj.n_words, 
																		 embed_dim = source_embed_dim, 
																		 hidden_size = source_hidden_size,
																		 rnn_layers = source_rnn_layers, 
																		 rnn_type = source_rnn_type).to(device);
																		 





		decoder = nnet_models.DecoderRNN(dataset_dict['train'].target_lang_obj.n_words, 
																								embed_dim = target_embed_dim, 
																								hidden_size = target_hidden_size, 
																								n_layers = target_rnn_layers, 
																								attention = attention).to(device)   





		encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.25,nesterov=True, momentum = 0.99)
		enc_scheduler = ReduceLROnPlateau(encoder_optimizer, min_lr=1e-4,  patience=0)
		decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.25,nesterov=True, momentum = 0.99)
		dec_scheduler = ReduceLROnPlateau(decoder_optimizer, min_lr=1e-4,  patience=0)





		criterion = nn.NLLLoss(ignore_index = global_variables.PAD_IDX)





		encoder, decoder, loss_hist, acc_hist = train_utilities.train_model(encoder_optimizer, decoder_optimizer, 
																								encoder, decoder, criterion,
																								attention, dataloader_dict, dataset_dict['train'].target_lang_obj, 
																								num_epochs = 8, rm = 0.95,
																								enc_scheduler = enc_scheduler, dec_scheduler = dec_scheduler)




		val_score = train_utilities.validation_function(encoder, decoder, dataloader_dict['val'], 
																											 dataset_dict['train'].target_lang_obj)




		def save_models(encoder, decoder, path):
				if not os.path.exists(path):
						os.makedirs(path)
				torch.save(encoder.state_dict(), os.path.join(path, 'encoder.pth'))
				torch.save(decoder.state_dict(), os.path.join(path, 'decoder.pth'))




		save_models(encoder, decoder, pth_saved_dir)




		encoder_optimizer = optim.Adam(encoder.parameters(), lr = 3e-4)
		decoder_optimizer = optim.Adam(decoder.parameters(), lr = 3e-4)

		enc_scheduler = ReduceLROnPlateau(encoder_optimizer, min_lr=1e-5,factor = 0.5,  patience=0)
		dec_scheduler = ReduceLROnPlateau(decoder_optimizer, min_lr=1e-5,factor = 0.5,  patience=0)




		encoder, decoder, loss_hist, acc_hist = train_utilities.train_model(encoder_optimizer, decoder_optimizer, 
																								encoder, decoder, criterion,
																								attention, dataloader_dict, dataset_dict['train'].target_lang_obj, 
																								num_epochs = 5, rm = 0.95,
																								enc_scheduler = enc_scheduler, dec_scheduler = dec_scheduler)




		new_val_score = train_utilities.validation_function(encoder, decoder, dataloader_dict['val'], 
																											 dataset_dict['train'].target_lang_obj)




		if new_val_score > val_score:
				save_models(encoder, decoder, pth_saved_dir)

		encoder.load_state_dict(torch.load( os.path.join(pth_saved_dir, 'encoder.pth') ))
		decoder.load_state_dict(torch.load( os.path.join(pth_saved_dir, 'decoder.pth') ))

		test_score = train_utilities.validation_function(encoder, decoder, dataloader_dict['test'], 
																											 dataset_dict['train'].target_lang_obj)

		np.savetxt( os.path.join(pth_saved_dir, 'scores.csv'), 
									np.array( [ np.max([new_val_score, val_score]), test_score ]))

