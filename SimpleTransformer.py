import numpy as np
import math

class SimpleTransformer:
    def __init__(self, d_model, n_heads, vocab_size):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = int(d_model / n_heads)
        self.vocab_size = vocab_size

        # Attention Weights
        self.Wq1 = np.random.rand(d_model, self.d_k)
        self.Wk1 = np.random.rand(d_model, self.d_k)
        self.Wv1 = np.random.rand(d_model, self.d_k)
        self.Wq2 = np.random.rand(d_model, self.d_k)
        self.Wk2 = np.random.rand(d_model, self.d_k)
        self.Wv2 = np.random.rand(d_model, self.d_k)
        self.Wo = np.random.rand(d_model, d_model)
        
        # FFN Weights
        self.W1 = np.random.rand(d_model, 8)
        self.W2 = np.random.rand(8, d_model)
        
        # Output Head
        self.W_vocab = np.random.rand(d_model, vocab_size)

        # Dictionary must match d_model size (4)
        self.emb = {
            "cat":  np.array([1.0, 0.2, 0.5, 0.2]),
            "dog":  np.array([0.9, 0.3, 1.0, 0.3]),
            "fish": np.array([0.1, 1.0, 0.2, 0.4]),
            "tree": np.array([0.2, 0.6, 0.1, 0.7])
        }

    def layer_norm(self, x):
        mu = np.mean(x, axis=1, keepdims=True)
        sigma = np.std(x, axis=1, keepdims=True)
        return (x - mu) / (sigma + 1e-6)
        
    def attention_head(self, x, mask, wq, wk, wv):
        Q = x @ wq
        K = x @ wk
        V = x @ wv
        scores = (Q @ K.T) / math.sqrt(self.d_k)
        masked_scores = scores + mask
        exp_val = np.exp(masked_scores)
        attention_weights = exp_val / exp_val.sum(axis=1, keepdims=True)
        return attention_weights @ V

    def forward(self, x):
        seq_len = len(x)
        mask = np.zeros((seq_len, seq_len))
        mask[np.triu_indices(seq_len, k=1)] = -1e9
        
        # Multi-Head Attention
        head1 = self.attention_head(x, mask, self.Wq1, self.Wk1, self.Wv1)
        head2 = self.attention_head(x, mask, self.Wq2, self.Wk2, self.Wv2)
        Z = np.concatenate((head1, head2), axis=1)
        multi_head_out = Z @ self.Wo

        # Add & Norm (Residual 1)
        ln1_out = self.layer_norm(multi_head_out + x)

        # Feed Forward Network
        ffn_hidden = ln1_out @ self.W1
        ffn_activated = np.maximum(0, ffn_hidden)
        ffn_out = ffn_activated @ self.W2

        # Add & Norm (Residual 2)
        ln2_out = self.layer_norm(ffn_out + ln1_out)

        # Output Projection
        last_word_vector = ln2_out[-1, :]
        logits = last_word_vector @ self.W_vocab
        return logits
        
    def generate(self, start_sentence, max_len=5, temperature=1.0): # Add it as an argument
        current_sentence = list(start_sentence)
        vocab_list = list(self.emb.keys())
        
        for _ in range(max_len):
            x = np.array([self.emb[word] for word in current_sentence])
            logits = self.forward(x)
            
            scaled_logits = logits / temperature 
            
            # probability math on the scaled versions
            exp_logits = np.exp(scaled_logits)
            probs = exp_logits / np.sum(exp_logits)
            
            next_word_idx = np.random.choice(len(vocab_list), p=probs)
            next_word = vocab_list[next_word_idx]
            current_sentence.append(next_word)
            
        return current_sentence

# exec
my_transformer = SimpleTransformer(d_model=4, n_heads=2, vocab_size=4)

# test gen
seed = ["tree"]
result = my_transformer.generate(seed, max_len=10, temperature=0.8)

print("-" * 30)
print(f"Seed Input: {seed}")
print(f"Final Generation: {' '.join(result)}")
print("-" * 30)