
from code import prepareData, normalizeString, EncoderRNN, AttnDecoderRNN, evaluate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os
import re
import random
lang1 = 'demotic'
lang2 = 'english'
source, target, pairs = prepareData(lang1, lang2)

randomize = random.choice(pairs)
print('random sentence {}'.format(randomize))


hidden_size = 512

#create encoder-decoder model
encoder = EncoderRNN(input_size, hidden_size, embed_size, num_layers)
decoder = AttnDecoderRNN(output_size, hidden_size, embed_size, num_layers)

# model = Seq2Seq(encoder, decoder, device).to(device)



encoder.load_state_dict(torch.load('./mytraining.pt'))
decoder.load_state_dict(torch.load('./mytraining.pt'))

evaluate(encoder, decoder, normalizeString(input()))
