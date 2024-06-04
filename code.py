import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()

print(digits.target[100])

from sklearn.cluster import KMeans

model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show();

new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,3.81,4.57,4.57,4.43,3.06,1.91,0.54,0.00,5.34,6.10,6.10,6.78,7.63,7.63,7.39,0.00,0.00,0.00,1.99,6.26,7.62,6.18,2.29,0.00,0.00,3.28,7.63,6.63,2.44,0.08,0.00,0.00,0.91,7.48,5.34,0.23,0.00,0.00,0.00,0.00,2.97,7.62,1.37,0.00,0.00,0.00,0.00,0.00,0.23,1.83,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.61,0.76,0.76,0.38,0.00,0.00,0.00,5.79,7.63,7.63,7.63,6.57,0.00,0.00,0.00,3.97,5.80,7.10,7.63,6.41,0.00,0.00,0.00,0.00,3.13,7.55,6.64,1.38,0.00,0.00,0.00,0.00,0.15,6.33,5.35,0.00,0.00,0.00,1.22,3.51,6.72,7.63,4.74,0.00,0.00,0.00,3.96,7.63,6.12,2.52,0.38,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.15,1.30,0.00,0.00,0.00,0.00,0.00,0.15,5.49,7.63,0.69,0.00,0.00,0.00,0.00,4.19,7.63,7.63,0.76,0.00,0.00,0.00,0.00,5.34,7.63,7.63,0.76,0.00,0.00,0.00,0.00,2.75,5.04,7.63,0.91,0.00,0.00,0.00,0.00,0.00,1.68,7.63,2.21,0.00,0.00,0.00,0.00,0.00,0.00,2.06,0.23,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.31,1.07,2.59,3.21,6.56,0.00,0.00,2.06,7.47,7.63,7.63,7.63,7.32,0.00,0.00,5.42,7.17,3.51,1.37,0.46,0.00,0.00,0.00,5.57,7.62,7.63,6.73,3.05,0.00,0.00,0.00,0.15,2.13,3.36,7.33,4.57,0.00,0.00,0.00,0.00,0.00,0.00,6.87,4.58,0.00,1.75,6.56,3.05,0.31,2.14,7.32,4.43,0.00,0.84,6.48,7.62,7.56,7.62,6.79,1.45,0.00]
])

new_labels = model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
