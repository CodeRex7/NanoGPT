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
        START_TOKEN, 
        END_TOKEN, 
        PADDING_TOKEN):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.vocab_size = len()
        self.embedding = nn.Embedding(self.vocab_size, model_embd)
        self.position_encoder = nn.Embedding(context_size, model_embd)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        
    def batch_tokenize(self, batch, start_token, end_token):
        return
    
    def forward(self, batch, start_token, end_token):
        idx = self.batch_tokenize(batch, start_token, end_token)
        B, T = idx.shape
        tok_emb = self.embedding(x)
        pos_emb = self.position_encoder(torch.arange(T, device=get_device()))
        x = self.dropout(tok_emb + pos_emb)
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
        START_TOKEN, 
        END_TOKEN, 
        PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = Sentence_Embedding(max_seq_length, model_embd, context_size, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.block = Sequential_Encoder(*[EncoderLayer(model_embd, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
        
    def forward(self, idx, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(idx, start_token, end_token)
        x = self.block(x, self_attention_mask)
        return x

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
        START_TOKEN, 
        END_TOKEN, 
        PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = Sentence_Embedding(max_seq_length, model_embd, context_size, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.block = Sequential_Decoder(*[DecodeLayer(model_embd, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
        
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
            START_TOKEN, 
            END_TOKEN, 
            PADDING_TOKEN
            ):
        super().__init__()
        self.device = get_device()
        self.encoder = Encoder(model_embd, ffn_hidden, num_heads, drop_prob, num_layers, max_seq_length, batch_size, context_size, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(model_embd, ffn_hidden, num_heads, drop_prob, num_layers, max_seq_length, batch_size, context_size, START_TOKEN, END_TOKEN, PADDING_TOKEN)
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