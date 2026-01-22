import numpy as np
import matplotlib.pyplot as plt

vecA = np.array([2, 1])
vecB = np.array([1, 3])

# Compute dot product
dotproduct = np.dot(vecA, vecB)
# Unit vector of A
unit_vector = vecA / np.linalg.norm(vecA)
# Projection length of B onto A
scalar_proj = dotproduct / np.linalg.norm(vecA)
# Projection vector
projection_vector = unit_vector * scalar_proj  # <-- must be 1D array of 2 numbers

# Plot all arrows
plt.quiver(0, 0, vecA[0], vecA[1], angles='xy', scale_units='xy', scale=1, color='blue', label='vecA')
plt.quiver(0, 0, vecB[0], vecB[1], angles='xy', scale_units='xy', scale=1, color='green', label='vecB')
plt.quiver(0, 0, projection_vector[0], projection_vector[1], angles='xy', scale_units='xy', scale=1, color='red', label='projection')

plt.axis('equal')
plt.xlim(0, 3)
plt.ylim(0, 4)
plt.legend()
plt.show()
