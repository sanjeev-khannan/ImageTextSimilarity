# ImageTextSimilarity
This is an ongoing experiment aimed at refining the model and improving its accuracy in predicting the similarity between images and captions.

I'm building and training a model to evaluate the similarity between images and their associated textual descriptions (captions) using the Flickr30k dataset. This model aims to learn a matching score between an image and a caption by distinguishing correct (true) pairs from incorrect (false) pairs.

### Dataset and Preprocessing

**Dataset**: The Flickr30k dataset contains images and multiple captions for each image. We use a CSV file (`results.csv`) where each row corresponds to an image name, a comment number, and a comment (caption).

**Text Preprocessing**:
- Converted captions to lowercase and removed non-alphanumeric characters using a preprocessing function.
- Tokenized the preprocessed captions to convert them into sequences of integers and padded them to a maximum length of 100 tokens.

**Image Preprocessing**:
- Defined a function to preprocess images by resizing them to 299x299 pixels and normalizing the pixel values to the range [0, 1].

### Data Generators

**Data Generator**:
- Created a data generator function to yield batches of image-caption pairs.
- For each image-caption pair, randomly decided whether to use the true caption (label=1) or a randomly chosen false caption (label=0) to create a balanced training dataset.

### Model Architecture

The model consists of two main components: an image processing branch and a text processing branch.

**Image Processing Branch**:
- Uses a series of Convolutional Neural Network (CNN) layers with batch normalization, ReLU activation, max pooling, and dropout layers to extract features from the input images.
- The final output is a global average pooling layer followed by a dense layer to get a 512-dimensional feature vector.

**Text Processing Branch**:
- Uses an embedding layer to convert tokenized caption sequences into dense vectors.
- Processes these vectors using three Bidirectional LSTM layers with dropout to capture sequential information from the captions.
- The output is a 256-dimensional feature vector.

**Combined Model**:
- Concatenates the image and text feature vectors.
- Passes the concatenated features through a dense layer with batch normalization and ReLU activation.
- The final layer is a dense layer with a sigmoid activation function to predict a similarity score (1 for true pairs and 0 for false pairs).

### Model Training

**Training Configuration**:
- Compiled the model with the Adam optimizer, binary cross-entropy loss, and accuracy as the metric.
- Defined a custom callback to save the model checkpoints every 250 batches.
- Loaded pre-trained weights if available.
- Trained the model using the data generator for one epoch and saved checkpoints.

### Evaluation

**Evaluation Process**:
- Used the validation data generator to iterate over the validation set and compute accuracy.
- Compared the model predictions to the true labels, converting predicted probabilities to binary values and calculating the proportion of correct predictions.
- Printed the overall accuracy on the validation set.

### Ongoing Work

This is an ongoing experiment aimed at refining the model and improving its accuracy in predicting the similarity between images and captions. Future steps may include further tuning of hyperparameters, increasing the training epochs, and possibly incorporating more sophisticated text and image processing techniques to enhance the model's performance.
