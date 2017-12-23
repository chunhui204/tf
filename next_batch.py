import numpy as np
#mnist_train_images: training data
#mnist_train_labels: training labels
#

end = 0
def next_batch(batch_size):
    global end
    global mnist_train_images
    global mnist_train_labels
    
    end = end+batch_size
    #if finish an epoch, permutate data randomly
    if end - batch_size > mnist_train_images.shape[0]:
        end = batch_size
        # not essential to specify a seed for permutation, np.random.permutation has different results every time
        permutation = np.random.permutation(mnist_train_images.shape[0])
        mnist_train_images = mnist_train_images[permutation,:]
        mnist_train_labels = mnist_train_labels[permutation,:]
    return mnist_train_images[end-batch_size:end,:], mnist_train_labels[end-batch_size:end,:]
