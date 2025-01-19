# Juho Peltomäki

## The Python code for training and predicting labels in MNIST-dataset
## utilizing the Naive Bayesian algorithm (with log-likelihood)
import matplotlib.pyplot as plt
import tensorflow as tf
from random import random
import numpy as np
# import pandas as pd

from scipy.stats import multivariate_normal


from sklearn.neighbors import KNeighborsClassifier  

# Original 
mnist = tf.keras.datasets.mnist
# New
# mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print the size of training and test data
print(f'x_train shape {x_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'x_test shape {x_test.shape}')
print(f'y_test shape {y_test.shape}')



x_train_new = x_train.reshape(x_train.shape[0], -1)
x_test_new = x_test.reshape(x_test.shape[0], -1)

print(f"x_train reshaped: {x_train_new.shape}")
print(f"x_test reshaped: {x_test_new.shape}")


# noise_scale = 1   # giving 66.87 - 66.92 % accuracy
#noise_scale = 0.1   # giving 61.66 - 61.67 % accuracy
noise_scale = 10     # giving  77.23 - 77.39  % accuracy

noisy_x_train = x_train_new + np.random.normal(loc=0.0, scale = noise_scale, size = x_train_new.shape)


noisy_x_train =  np.clip(noisy_x_train, 0, 255)


# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(x_train_new, y_train)
# y_pred = knn.predict(x_test_new)


# for i in range(x_test.shape[0]):
#     break
#     # Show some images randomly
#     if random() > 0.999:
#         plt.figure(1)
#         plt.clf()
#         plt.imshow(x_test[i], cmap='gray_r')
#         plt.title(f"Image {i} label num {y_test[i]} predicted {0}")
#         plt.pause(1)
        
        

n = 10
num_features = x_train_new.shape[1]


mean_vectors = []
variance_vectors = []


little_value = 0.01

for k in range(n):
    
    class_k_data = noisy_x_train[y_train == k]
    
    mean_vector = np.mean(class_k_data, axis=0)
    mean_vectors.append(mean_vector)
    
    
    variance_vector = np.var(class_k_data, axis=0) + little_value
    variance_vectors.append(variance_vector)
    
    
mean_vectors = np.array(mean_vectors)
variance_vector = np.array(variance_vectors)



## actual algorithm implementation

def compute_log_likelihood(x, mean_vectors, variance_vectors):


    log_likelihoods = []
    
    for k in range(n):
        mu_k = mean_vectors[k]
        var_k = variance_vectors[k]
        
        
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var_k)) - 0.5 * np.sum(((x - mu_k) ** 2) / var_k)
        log_likelihoods.append(log_likelihood)
    
    return np.array(log_likelihoods)

y_pred = []

for pixel in x_test_new:
    log_likelihoods = compute_log_likelihood(pixel, mean_vectors, variance_vectors)
    predicted_class = np.argmax(log_likelihoods)  
    y_pred.append(predicted_class)


def acc(pred, gt):
    
    
    zeros = np.sum((gt - pred) == 0)
    accuracy = zeros / 10000
  #  print(accuracy, "as decimal ")
    return accuracy 


accuracy = acc(y_test, y_pred)

print("----------------------------")
print(f"Classification accuracy: {accuracy * 100:.2f}% (with noise value of {noise_scale})")

# accu = acc(y_test, y_pred)

random_predictions = np.random.randint(0, 10, size=y_test.shape)

# calculating accuracy between random predictions and true test labels
accuracy = acc(random_predictions, y_test)
print(f"Accuracy of random predictions: {accuracy*100:.2f}%")


