import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniLLM(nn.Module):
    def __init__(self, vocab_size=4, d_model=4, n_heads=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Multi-Head Attention parts
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 8),
            nn.ReLU(),
            nn.Linear(8, d_model)
        )
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        # x is now a list of word indices like [0, 3]
        x = self.embedding(x) # Turns [0, 3] into a matrix of 4D vectors
        
        # 1. Attention logic
        # PyTorch can do the Q, K, V math in one go (I did not do that yet)
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        
        # 2. Add & Norm
        attention_out = self.Wo(v) # Simplified
        x = self.ln1(x + attention_out)
        
        # 3. FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return self.output_head(x)
    
    def generate(self, start_indices, max_len=5, temperature=1.0):
        # start_indices should be a list of numbers like [3]
        current_seq = torch.tensor(start_indices).unsqueeze(0) # Make it (1, seq_len)
        
        for _ in range(max_len):
            logits = self.forward(current_seq) # Get logits
            last_token_logits = logits[0, -1, :] / temperature # Take last word and apply temp
            
            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
            
        return current_seq.squeeze().tolist()

# exec
word2idx = {"cat": 0, "dog": 1, "fish": 2, "tree": 3}
idx2word = {i: w for w, i in word2idx.items()}

my_transformer = MiniLLM(vocab_size=4, d_model=4)
seed_indices = [word2idx["tree"]]
output_indices = my_transformer.generate(seed_indices, max_len=5)

print("Generated:", [idx2word[i] for i in output_indices])