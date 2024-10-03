# Dog-Breed-Prediction

This project focuses on predicting dog breeds from images using deep learning techniques. The model is trained to classify three specific dog breeds: 'Scottish Deerhound', 'Maltese Dog', and 'Bernese Mountain Dog'. The architecture used for this task is a Convolutional Neural Network (CNN) built with Keras, optimized for image classification.

Model Architecture
The model consists of the following layers:

Convolutional Layers:

Layer 1: 64 filters, kernel size of (5x5), ReLU activation, and input shape of (224,224,3) for the input images.
Layer 2: 32 filters, kernel size of (3x3), ReLU activation, L2 regularization.
Layer 3: 16 filters, kernel size of (7x7), ReLU activation, L2 regularization.
Layer 4: 8 filters, kernel size of (5x5), ReLU activation, L2 regularization.
Each convolutional layer is followed by a MaxPooling layer with a pool size of (2x2) to reduce the spatial dimensions of the feature maps.

Fully Connected Layers:

Flatten layer to convert the 2D feature maps into a 1D feature vector.
Dense Layer 1: 128 neurons, ReLU activation, L2 regularization.
Dense Layer 2: 64 neurons, ReLU activation, L2 regularization.
Output Layer: Number of neurons equal to the number of classes (in this case, 3), using the softmax activation function for multiclass classification.
Optimizer and Loss:

The model uses Adam optimizer with a learning rate of 0.0001.
The loss function is categorical crossentropy for multiclass classification.
Accuracy is used as the evaluation metric.
