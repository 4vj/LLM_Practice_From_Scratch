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
X = np.matrix(seq_len, d) # seq_len means the amount of words in the array
Query_matrix = np.array([[3.23, 4, 2], [2, 1, 3]])
Key_matrix = np.array([[1, 3, 2], [3, 4, 0.6]])
value_matrix = np.array([[2.6, 1, 0.5], [3, 4, 6]])
Q = Query_matrix
K = Key_matrix
V = value_matrix
Scores = Q @ K.T
print(Scores)