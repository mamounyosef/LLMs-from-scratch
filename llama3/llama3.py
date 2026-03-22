import torch
import torch.nn as nn
from torch.nn import functional as F

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim, num_heads, head_size, rope_theta, dropout_rate):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_size = head_size
        self.rope_theta = rope_theta

        self.kqv = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        B, T, D = x.shape # x (input) of size (batch, seq_len, embedding_dim)

        k, q, v = self.kqv(x).split(self.embedding_dim, dim=-1)
        print(f"k.shape: {k.shape}")
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # finally its (B, num_heads, seq_len, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, seq_len, head_size)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, seq_len, head_size)

        # RoPE Implementation
        m = torch.arange(T, device=x.device) # (seq_len)
        i = torch.arange(self.head_size//2, device=x.device) # (head_size/2)

        theta = torch.pow(self.rope_theta, (-2*i)/self.head_size) # (head_size/2,)
        m_theta = torch.outer(m, theta) # (seq_len, head_size/2)
    
        sin = torch.sin(m_theta) # (seq_len, head_size/2)
        cos = torch.cos(m_theta) # (seq_len, head_size/2)

        q_even = q[:, :, :, ::2] # (B, seq_len, head_size/2)
        q_odd = q[:, :, :, 1::2] # (B, seq_len, head_size/2)

        q_1 = q_even * cos - q_odd * sin # (B, seq_len, head_size/2)
        q_2 = q_even * sin + q_odd * cos # (B, seq_len, head_size/2)
        q_rotated = torch.stack([q_1, q_2], dim=-1).reshape(q.shape) # (B, seq_len, head_size)

        k_even = k[:, :, :, ::2] # (B, seq_len, head_size/2)
        k_odd = k[:, :, :, 1::2] # (B, seq_len, head_size/2)

        k_1 = k_even * cos - k_odd * sin # (B, seq_len, head_size/2)
        k_2 = k_even * sin + k_odd * cos # (B, seq_len, head_size/2)
        k_rotated = torch.stack([k_1, k_2], dim=-1).reshape(k.shape) # (B, seq_len, head_size)

        # Flash Attention with causal mask
        attn_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        x = F.scaled_dot_product_attention(
            q_rotated, k_rotated, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False  # we provide explicit mask here
        )
        x = x.transpose(1, 2).contiguous().view(B, T, D)
        x = self.proj(x)
        x = self.dropout(x)
        return x
    
class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # x: (B, T, D)
        self.W1 = nn.Linear(self.embedding_dim, self.hidden_dim, bias=False) # (D, H) → (B, T, H)
        self.W3 = nn.Linear(self.embedding_dim, self.hidden_dim, bias=False) # (D, H) → (B, T, H)
        self.W2 = nn.Linear(self.hidden_dim, self.embedding_dim, bias=False) # (H, D) → (B, T, D)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.silu(self.W1(x)) * self.W3(x) # element-wise multiply: (B, T, H)
        x = self.dropout(self.W2(x))
        return 



class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads, head_size, rope_theta, dropout):

        super().__init__()
        self.mha = MultiHeadAttention(embedding_dim, num_heads, head_size, rope_theta, dropout)
        self.ffnn = FeedForwardNN(embedding_dim, hidden_dim, dropout)
        self.norm1 = nn.RMSNorm(embedding_dim)      
        self.norm2 = nn.RMSNorm(embedding_dim)    

    def forward(self, x):

        x = self.norm1(x)
        x = x + self.mha(x) # the skip connection

        x = self.norm2(x)
        x = x + self.ffnn(x) # the skip connection

        return x 

class llama(nn.Module):
    def __init__(self, n_layers=32, embedding_dim=4096, hidden_dim=14_336, num_heads=32, max_seq_len=1024, vocab_size=128_000, rope_theta=500_000, dropout_rate=0.1):

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
        self.blocks = nn.ModuleList([DecoderBlock(embedding_dim, hidden_dim, num_heads, self.hyperparamiters['head_size'], rope_theta, dropout_rate) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(embedding_dim)
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