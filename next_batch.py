import numpy as np
#mnist_train_images: training data
#mnist_train_labels: training labels
#

start = 0
def next_batch(batch_size):
    global start
    global mnist_train_images
    global mnist_train_labels
    
    start = start+batch_size
    if start >= mnist_train_images.shape[0]:
        start = 0
        permutation = np.random.permutation(mnist_train_images.shape[0])
        mnist_train_images = mnist_train_images[permutation,:]
        mnist_train_labels = mnist_train_labels[permutation,:]
    return mnist_train_images[start:start+batch_size,:], mnist_train_labels[start:start+batch_size,:]
