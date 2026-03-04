import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):

    def __init__(self, embedding_dim, head_size, dropout):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.head_size = head_size

        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # x (input) of size (batch, seq_len, embedding_dim)

        seq_len = x.shape[1]

        k = self.key(x) # (B, seq_len, head_size)
        q = self.query(x) # (B, seq_len, head_size)
        v = self.value(x) # (B, seq_len, head_size)

        # Flash Attention with causal mask
        attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False  # we provide explicit mask
        )

        return out

class MultiHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim, num_heads, head_size, dropout_rate):

        super().__init__()
        self.heads = nn.ModuleList([Head(embedding_dim, head_size, dropout_rate) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x
    
class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(   
            nn.Linear(self.embedding_dim, 4 * self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)



class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, head_size, dropout):

        super().__init__()
        self.mha = MultiHeadAttention(embedding_dim, num_heads, head_size, dropout)
        self.ffnn = FeedForwardNN(embedding_dim, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)      
        self.norm2 = nn.LayerNorm(embedding_dim)      
        
    def forward(self, x):

        x = self.norm1(x)
        x = x + self.mha(x) # the skip connection

        x = self.norm2(x)
        x = x + self.ffnn(x) # the skip connection

        return x 

class GPT(nn.Module):
    def __init__(self, n_layers = 12, embedding_dim = 768, num_heads = 12, max_seq_len = 1024, vocab_size = 50432, dropout_rate = 0.1):

        super().__init__()
        self.hyperparamiters = {
            'n_layers': n_layers,
            'embedding_dim': embedding_dim, # also called d_model
            'num_heads': num_heads,
            'head_size': embedding_dim // num_heads, # automatically calculated
            'max_seq_len': max_seq_len,
            'vocab_size': vocab_size # (context size)
        }

        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding_table = nn.Embedding(max_seq_len, embedding_dim)
        self.blocks = nn.ModuleList([DecoderBlock(embedding_dim, num_heads, self.hyperparamiters['head_size'], dropout_rate) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier_layer = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

        self.classifier_layer.weight = self.embedding_table.weight # tying the weights together

    def forward(self, ids, targets=None):
        b, seq_len = ids.shape # ids is (batch_size, seq_len)

        embeddings = self.embedding_table(ids) # (batch_size, seq_len, embedding_dim)
        pos_embeddings = self.pos_embedding_table(torch.arange(seq_len, device=device))

        x = embeddings + pos_embeddings # (batch_size, seq_len, embedding_dim)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        logits = self.classifier_layer(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss