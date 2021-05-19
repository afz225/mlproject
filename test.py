
# coding=utf-8
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

encoder1.load_state_dict(torch.load('./experiments/standrew/myencoder.pt', map_location=torch.device('cpu')))
attn_decoder1.load_state_dict(torch.load('./experiments/standrew/mydecoder.pt', map_location=torch.device('cpu')))

encoder2.load_state_dict(torch.load('./experiments/standrew_and_beinlich/myencoder.pt', map_location=torch.device('cpu')))
attn_decoder2.load_state_dict(torch.load('./experiments/standrew_and_beinlich/mydecoder.pt', map_location=torch.device('cpu')))

encoder3.load_state_dict(torch.load('./experiments/translated_german_and_beinlich/myencoder.pt', map_location=torch.device('cpu')))
attn_decoder3.load_state_dict(torch.load('./experiments/translated_german_and_beinlich/mydecoder.pt', map_location=torch.device('cpu')))

# print(' '.join(evaluate(encoder1, attn_decoder1, normalizeString(input('input egyptian here: ')))[0]))



from nltk.translate.bleu_score import sentence_bleu

reference = []
candidate1 = []
candidate2 = []
candidate3 = []

reference.append(normalizeString('next to Anubis, Lord of the Holy Land, the chief of artworkers, the scribe and sculptor Irtysen. He says: I').split(' '))

# candidate1.append(evaluate(encoder1, attn_decoder1, normalizeString('xr Inpw nb tA Dsr (i)m(y)-r(A) Hmw.t sS qs.ty Ir.ty=sn Dd(=f) iw rx'))[0].split(' '))
# candidate2.append(evaluate(encoder2, attn_decoder2, normalizeString('xr Inpw nb tA Dsr (i)m(y)-r(A) Hmw.t sS qs.ty Ir.ty=sn Dd(=f) iw rx'))[0].split(' '))
# candidate3.append(evaluate(encoder3, attn_decoder3, normalizeString('xr Inpw nb tA Dsr (i)m(y)-r(A) Hmw.t sS qs.ty Ir.ty=sn Dd(=f) iw rx'.replace('A', 'ꜣ').replace('j', 'i҆').replace('a', 'ꜥ').replace('H', 'ḥ').replace('x', 'ḫ').replace('X', 'ẖ').replace('S', 'š').replace('q', 'ḳ').replace('T', 'ṯ').replace('D','ḏ').replace('jj', 'y'))[0].split(' '))



print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))


# print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate1, weights=(1, 0, 0, 0)))
# print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate1, weights=(0.5, 0.5, 0, 0)))
# print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate1, weights=(0.33, 0.33, 0.33, 0)))
# print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate1, weights=(0.25, 0.25, 0.25, 0.25)))


# print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate2, weights=(1, 0, 0, 0)))
# print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate2, weights=(0.5, 0.5, 0, 0)))
# print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate2, weights=(0.33, 0.33, 0.33, 0)))
# print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate2, weights=(0.25, 0.25, 0.25, 0.25)))


# print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate3, weights=(1, 0, 0, 0)))
# print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate3, weights=(0.5, 0.5, 0, 0)))
# print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate3, weights=(0.33, 0.33, 0.33, 0)))
# print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate3, weights=(0.25, 0.25, 0.25, 0.25)))

