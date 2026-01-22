import matplotlib
matplotlib.use('qtagg') 
import matplotlib.pyplot as plt
import numpy as np
import math
emb = {
  "cat":  np.array([1.0, 0.2, 0.5, 0.2]),   
  "dog":  np.array([0.9, 0.3, 1.0, 0.3]),
  "fish": np.array([0.1, 1.0, 0.2, 0.4]),   
  "tree": np.array([0.2, 0.6, 0.1, 0.7])    
}
Sentence = ["tree", "cat", "fish"]
seq_len = len(Sentence)
mask = np.zeros((seq_len, seq_len))
mask[np.triu_indices(seq_len, k=1)] = -1e9 #"infinity"
d_model = 4 # because of this simple example the amount of dimensions is only 2 (2 numbers in a array)
n_heads = 2
d_k = d_model / n_heads
print(d_k)
X = np.zeros((seq_len, d_model)) # seq_len means the amount of words in the array
for i, word in enumerate(Sentence):
    X[i] = emb[word]
# Define random weight matrices for Query, Key, and Value for each head
Wq1 = np.random.rand(d_model, int(d_k))
Wk1 = np.random.rand(d_model, int(d_k))
Wv1 = np.random.rand(d_model, int(d_k))

Q1 = X @ Wq1
K1 = X @ Wk1
V1 = X @ Wv1

print("Q1:", Q1)
print("K1:", K1)
print("V1:", V1)

Wq2 = np.random.rand(d_model, int(d_k))
Wk2 = np.random.rand(d_model, int(d_k))
Wv2 = np.random.rand(d_model, int(d_k))

Q2 = X @ Wq2
K2 = X @ Wk2
V2 = X @ Wv2

print("Q2:", Q2)
print("K2:", K2)
print("V2:", V2)

masked_scores1 = Q1 @ K1.T / math.sqrt(d_k) + mask
exp_val1 = np.exp(masked_scores1)
total_pool1 = exp_val1.sum(axis=1, keepdims=True)
attention1 = exp_val1 / total_pool1
head1_out = attention1 @ V1
print("attention1", attention1)
print("head1_out", head1_out)

masked_scores2 = Q2 @ K2.T / math.sqrt(d_k) + mask
exp_val2 = np.exp(masked_scores2)
total_pool2 = exp_val2.sum(axis=1, keepdims=True)
attention2 = exp_val2 / total_pool2
head2_out = attention2 @ V2
print("attention2", attention2)
print("head2_out", head2_out)

Z = np.concatenate((head1_out, head2_out), axis=1)
print("Z: ", Z)
Wo = np.random.rand(d_model, d_model)
MultiHead_Out = Z @ Wo
print("MultiHead_Out: ", MultiHead_Out)

Added_out = X + MultiHead_Out # Residual
mu = np.mean(Added_out, axis=1, keepdims=True)
sigma = np.std(Added_out, axis=1, keepdims=True)

LayerNorm = (Added_out - mu) / (sigma + 0.000001)

print("test:", np.mean(LayerNorm, axis=1))

W1 = np.random.rand(4, 8)
W2 = np.random.rand(8, 4)

Linear1 = LayerNorm @ W1
ReLu = np.maximum(0, Linear1)
Linear2 = ReLu @ W2
Added_Final = LayerNorm + Linear2
Mu_Final = np.mean(Added_Final, axis=1, keepdims=True)
Sigma_Final = np.std(Added_Final, axis=1, keepdims=True)
Final_result = (Added_Final - Mu_Final) / (Sigma_Final + 0.000001)

print("Final_result: ", Final_result)
print("Standard Deviation Test:", np.std(Final_result, axis=1))

W_vocab = np.random.rand(4, 4)
winner = np.argmax(Final_result)
last_word_vector = Final_result[-1, :]
Logits = last_word_vector @ W_vocab
predicted_index = np.argmax(Logits)
vocab_list = list(emb.keys()) # ["cat", "dog", "fish", "tree"]
predicted_word = vocab_list[predicted_index]

print(f"The model predicts the next word is: {predicted_word}")

def plot_final_view(original, transformed, attn, labels):
    # We create 3 subplots now
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Original Fingerprints
    ax1.imshow(original, cmap='viridis')
    ax1.set_title("Original (X)")
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    
    # 2. Transformed Fingerprints
    ax2.imshow(transformed, cmap='viridis')
    ax2.set_title("Final Result")
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    
    # 3. MASKED ATTENTION
    im3 = ax3.imshow(attn, cmap='magma')
    ax3.set_title("Causal Attention (Head 1)")
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels)
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels)
    fig.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.show()

# Call the new function
plot_final_view(X, Final_result, attention1, Sentence)