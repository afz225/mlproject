from code import prepareData, hidden_size,EncoderRNN, AttnDecoderRNN, evaluate,normalizeString, device

input_lang, output_lang, pairs = prepareData('demotic', 'english')

encoder2 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder2 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.05)

evaluate(encoder2, attn_decoder2, normalizeString(input('enter demotic input: ')))