import numpy as np
import matplotlib.pyplot as plt
import math
emb = {
  "cat":  np.array([1.0, 0.2]),   
  "dog":  np.array([0.9, 0.3]),
  "fish": np.array([0.1, 1.0]),  
  "tree": np.array([0.2, 0.6])    
}
Sentence = ["tree", "cat", "fish"]
seq_len = len(Sentence)
d = 2 # because of this simple example the amount of dimensions is only 2 (2 numbers in a array)
X = np.zeros((seq_len, d)) # seq_len means the amount of words in the array

for i, word in enumerate(Sentence):
    X[i] = emb[word]

Wq = np.random.randn(d, d)
Wk = np.random.randn(d, d)
Wv = np.random.randn(d, d)
Q = X @ Wq.T
K = X @ Wk.T
V = X @ Wv.T

scores = Q @ K.T
scaled_scores = scores / math.sqrt(d)
attention = np.zeros((seq_len, d))
weights = np.exp(scaled_scores) / np.exp(scaled_scores).sum(axis=1, keepdims=True)
attention = weights @ V

plt.imshow(weights, cmap="Blues", interpolation='nearest')
plt.title("Heatmap of Attention Weights")
plt.xlabel("Keys")
plt.ylabel("Queries")
plt.xticks(range(seq_len), Sentence)
plt.yticks(range(seq_len), Sentence)
plt.colorbar()
plt.show()

print("X shape:", X.shape)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
print("Attention:\n", attention)
#for i, word in enumerate(Sentence):
#    softmax = np.exp(scaled_scores[i]) / np.exp(scaled_scores[i]).sum()
#weights = np.exp(scaled_scores[i]) / np.exp(scaled_scores[i]).sum()
#attention_i = weights @ V