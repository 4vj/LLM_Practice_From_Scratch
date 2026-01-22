import numpy as np
import matplotlib.pyplot as plt
import math
Xx = 2
Xy = 1
Xbx = 1
Xby = 3
vecA = np.array([Xx, Xy])
vecB = np.array([Xbx, Xby])
#unit_vector = np.linalg.norm(vecA) / 0
np.dot(vecA, vecB, out=None)
dotproduct = (vecA[0] * vecB[0]) + (vecA[1] * vecB[1])
print(dotproduct)
unit_vector = vecA / np.linalg.norm(vecA)
scalar_proj = np.dot(vecA, vecB) / np.linalg.norm(vecA)
projection_vector = unit_vector * scalar_proj
#proj_vector = np.array(projection_vector[0], projection_vector[1])
#print (projection_vector)
#print (proj_vector)
#plt.quiver(0, 0, unit_vector, projection_vector, angles='xy', scale_units='xy', scale=1)
plt.quiver(0, 0, Xx, Xy, angles='xy', scale_units='xy', scale=1)
plt.quiver(0, 0, Xbx, Xby, angles='xy', scale_units='xy', scale=1)
plt.quiver(0, 0, projection_vector[0], projection_vector[1], color='red')
#plt.axis("equal")
#plt.xlim(0, 3)
#plt.ylim(0, 4)
plt.axis([0, max(Xx, Xbx)+1, 0, max(Xy, Xby)+1])
#plt.axis(0, Xx, 0, Xby )
plt.show()