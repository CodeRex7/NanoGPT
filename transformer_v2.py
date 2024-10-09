import numpy as np
import torch
import math
from torch import nn

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Sentence_Embedding(nn.module):
    def __init__(
        self,
        max_seq_length,
        model_embd,
        context_size,
        language_to_index,
        START_TOKEN, 
        END_TOKEN, 
        PADDING_TOKEN):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.vocab_size = len()
        self.embedding = nn.Embedding(self.vocab_size, model_embd)
        self.position_encoder = nn.Embedding(context_size, model_embd)
        self.language_to_index = language_to_index
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self, batch, start_token, end_token):
        idx = self.batch_tokenize(batch, start_token, end_token)
        B, T = idx.shape
        tok_emb = self.embedding(x)
        pos_emb = self.position_encoder(torch.arange(T, device=get_device()))
        x = self.dropout(tok_emb + pos_emb)
        return x

class LayerNormalization(nn.module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        # calculate the forward pass
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        xmean = inputs.mean(dim=dims, keepdim=True) # batch mean
        xvar = inputs.var(dim=dims, keepdim=True) # batch variance
        xhat = (inputs - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_embd, ffn_hidden, drop_prob):
        super(PositionwiseFeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(model_embd, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, model_embd),
            nn.Dropout(drop_prob),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Sequential_Encoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, model_embd, ffn_hidden, num_heads, drop_prob, vocab_size):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_embd=model_embd, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[model_embd])
        self.linear1 = nn.Linear(model_embd, vocab_size)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(model_embd=model_embd, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[model_embd])
        self.linear2 = nn.Linear(model_embd, vocab_size)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        residual_x = x.clone()
        x = self.norm1(x)
        x = self.attention(x, mask=self_attention_mask)
        x = x + residual_x
        x = self.linear1(x)
        x = self.dropout1(x)
        residual_x = x.clone()
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual_x
        x = self.linear2(x)
        x = self.dropout2(x)
        return x
    
class Encoder(nn.module):
    def __init__(
        self,
        model_embd, 
        ffn_hidden, 
        num_heads, 
        drop_prob, 
        num_layers, 
        max_seq_length, 
        batch_size, 
        context_size,
        vocab_size,
        english_to_index, 
        START_TOKEN, 
        END_TOKEN, 
        PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = Sentence_Embedding(max_seq_length, model_embd, context_size, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.block = Sequential_Encoder(*[EncoderLayer(model_embd, ffn_hidden, num_heads, drop_prob, vocab_size) for _ in range(num_layers)])
        
    def forward(self, idx, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(idx, start_token, end_token)
        x = self.block(x, self_attention_mask)
        return x

class Sequential_Decoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y
    
class DecoderLayer(nn.Module):
    def __init__(self, model_embd, ffn_hidden, num_heads, drop_prob, vocab_size):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_embd=model_embd, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[model_embd])
        self.linear1 = nn.Linear(model_embd, vocab_size)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(model_embd=model_embd, num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=[model_embd])
        self.linear2 = nn.Linear(model_embd, vocab_size)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(model_embd=model_embd, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=[model_embd])
        self.linear3 = nn.Linear(model_embd, vocab_size)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        residual_y = y.clone()
        y = self.layer_norm1(y)
        y = self.self_attention(y, mask=self_attention_mask)
        y = y + residual_y
        y = self.linear1(y)
        y = self.dropout1(y)

        residual_y = y.clone()
        y = self.layer_norm2(y)
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = y + residual_y
        y = self.linear2(y)
        y = self.dropout2(y)

        residual_y = y.clone()
        y = self.layer_norm3(y)
        y = self.ffn(y)
        y = y + residual_y
        y = self.linear3(y)
        y = self.dropout3(y)
        return y
     
class Decoder(nn.module):
    def __init__(
        self,
        model_embd, 
        ffn_hidden, 
        num_heads, 
        drop_prob, 
        num_layers, 
        max_seq_length, 
        batch_size, 
        context_size,
        vocab_size,
        portugese_to_index, 
        START_TOKEN, 
        END_TOKEN, 
        PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = Sentence_Embedding(max_seq_length, model_embd, context_size, portugese_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.block = Sequential_Decoder(*[DecoderLayer(model_embd, ffn_hidden, num_heads, drop_prob, vocab_size) for _ in range(num_layers)])
        
    def forward(self, x, idy, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(idy, start_token, end_token)
        y = self.block(x, y, self_attention_mask, cross_attention_mask)
        return y

class Transformer(nn.module):
    def __init__(
            self,
            model_embd,
            ffn_hidden,
            num_heads,
            drop_prob,
            num_layers,
            max_seq_length,
            vocab_size,
            batch_size,
            context_size,
            max_iters,
            eval_iters,
            learning_rate,
            english_to_index,
            portugese_to_index,
            START_TOKEN, 
            END_TOKEN, 
            PADDING_TOKEN
            ):
        super().__init__()
        self.device = get_device()
        self.encoder = Encoder(model_embd, ffn_hidden, num_heads, drop_prob, num_layers, max_seq_length, english_to_index, batch_size, context_size, vocab_size, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(model_embd, ffn_hidden, num_heads, drop_prob, num_layers, max_seq_length, portugese_to_index, batch_size, context_size, vocab_size, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layerNorm = nn.LayerNorm(model_embd)
        self.linear = nn.linear(model_embd, vocab_size)

    def forward(
            self,
            idx,
            idy,
            encoder_self_attention_mask=None, 
            decoder_self_attention_mask=None, 
            decoder_cross_attention_mask=None,
            enc_start_token=False,
            enc_end_token=False,
            dec_start_token=False,
            dec_end_token=False):
        x = self.encoder(idx, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, idy, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.layerNorm(out)
        out = self.linear(out)
        return out


with open('dataset/english.txt', 'r', encoding='utf-8') as f:
    english_file = f.read()

# here are all the unique characters that occur in this text
eng_chars = sorted(list(set(english_file)))
eng_vocab_size = len(eng_chars)

# create a mapping from characters to integers
eng_stoi = { ch:i for i,ch in enumerate(eng_chars) }
eng_itos = { i:ch for i,ch in enumerate(eng_chars) }
eng_encode = lambda s: [eng_stoi[c] for c in s] # encoder: take a string, output a list of integers
eng_decode = lambda l: ''.join([eng_itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
eng_data = torch.tensor(eng_encode(english_file), dtype=torch.long)
n = int(0.9*len(eng_data)) # first 90% will be train, rest val
train_data_eng = eng_data[:n]
val_data_eng = eng_data[n:]
 
with open('dataset/portugese.txt', 'r', encoding='utf-8') as f:
    portugese_file = f.read()
    
# here are all the unique characters that occur in this text
por_chars = sorted(list(set(portugese_file)))
por_vocab_size = len(por_chars)

# create a mapping from characters to integers
por_stoi = { ch:i for i,ch in enumerate(por_chars) }
por_itos = { i:ch for i,ch in enumerate(por_chars) }
por_encode = lambda s: [por_stoi[c] for c in s] # encoder: take a string, output a list of integers
por_decode = lambda l: ''.join([por_itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
por_data = torch.tensor(por_encode(portugese_file), dtype=torch.long)
n = int(0.9*len(por_data)) # first 90% will be train, rest val
train_data_por = por_data[:n]
val_data_por = por_data[n:]

model_embd = 512
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
batch_size = 16
context_size = 32
max_iters = 5000
eval_iters = 100
learning_rate = 1e-3
START_TOKEN = ''
END_TOKEN = ''
PADDING_TOKEN = ''

transformer = Transformer(model_embd, 
                          ffn_hidden,
                          num_heads, 
                          drop_prob, 
                          num_layers, 
                          max_sequence_length,
                          por_vocab_size,
                          max_iters,
                          eval_iters,
                          learning_rate,
                          eng_encode,
                          por_encode,
                          START_TOKEN, 
                          END_TOKEN, 
                          PADDING_TOKEN)