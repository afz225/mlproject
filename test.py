
from code import prepareData, normalizeString, EncoderRNN, AttnDecoderRNN, evaluate, input_lang, output_lang, device

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os
import re
import random
# lang1 = 'egyptian'
# lang2 = 'english'
# source, target, pairs = prepareData(lang1, lang2)



# hidden_size 


hidden_size = 512
encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.05)

encoder1.load_state_dict(torch.load('./myencoder.pt', map_location=torch.device('cpu')))
attn_decoder1.load_state_dict(torch.load('./mydecoder.pt', map_location=torch.device('cpu')))

# print(' '.join(evaluate(encoder1, attn_decoder1, normalizeString(input('input egyptian here: ')))[0]))

import sacrebleu
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang='en')

reference = []
candidate = []

file = open('./training/egyptian-english.txt', 'r')

for i in range(1000):
	line = file.readline().strip().split(';')
	
	if line[0] != '' and line[1] != '':
		reference.append(md.detokenize(normalizeString(line[1]).strip().split(' ')))
		temp = ' '.join(evaluate(encoder1, attn_decoder1, normalizeString(line[0]))[0])
		temp = temp.strip()
		temp = temp.split(' ')
		temp.pop()
		candidate.append(md.detokenize(temp))



file.close()

print(reference[501])
print(candidate[501])

print(len(reference))
print(len(candidate))
print('corpus_bleu: %f' % sacrebleu.corpus_bleu(reference, candidate))
