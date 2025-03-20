# OpinionSense: Unveiling the Emotions Behind Words

![image](https://github.com/MadhumithaKolkar/OpinionSense_RNN/assets/54811937/f5341b39-bc5c-482f-a338-06534b37374f)


This project builds a sentiment analysis model using a Recurrent Neural Network (RNN) to classify movie reviews as positive or negative. Here's a breakdown of its key steps:

## 1. Data Acquisition and Preprocessing:

Downloads the IMDB movie review dataset (aclImdb_v1.tar.gz).
Splits the data into training and validation sets (80%/20%) with a fixed random seed for reproducibility.
Preprocesses the text data:
Converts to lowercase.
Removes HTML tags (<br>).
Removes punctuation characters.

## 2. Text Vectorization:

Uses a TextVectorization layer to transform text into numerical representations suitable for the RNN:
Applies the custom preprocessing steps.
Splits text into individual words (tokens).
Builds a vocabulary of the vocab_size most frequent words (10,000 in this case).
Assigns a unique integer ID to each word based on the vocabulary.
Sets a maximum sequence length (sequence_length is 100 words) and pads shorter sequences.

## 3. Embedding Layer (Trainable):

Employs an Embedding layer to map integer IDs (from text vectorization) to dense embedding vectors:
Creates a random embedding matrix with vocab_size rows (one for each word) and embedding_dim columns (16 in this case).
Each row represents a word's embedding vector in the embedding space.
Unlike pre-trained models (Word2Vec, GloVe), this code does not load external embeddings.

## 4. Model Architecture (RNN):

Defines a Sequential model with the following layers:
The TextVectorization layer from step 2.
Embedding layer (described above).
SimpleRNN(8): A single-layer RNN with 8 hidden units to process the sequence of word embeddings.
Dense(1, activation='sigmoid'): The output layer with one unit and sigmoid activation for binary classification (positive or negative sentiment).

## 5. Model Training:

Trains the model on the training dataset (train_ds) with validation data (val_ds) for a specific number of epochs (15 in this case).
During training, the model learns to adjust the weights in the hidden layers and the embedding matrix.
The embedding vectors gradually capture word relationships relevant to sentiment analysis based on the training data.

## 6. User Interaction (Optional):

Allows users to enter a review text.
Predicts the sentiment (probability of being positive) for the entered review using the trained model.

# Key Points:

This model trains its own word embeddings from scratch instead of using pre-trained ones.
This approach can be suitable for smaller datasets or when domain-specific word relationships are important.
Consider exploring pre-trained embeddings or more complex RNN architectures (LSTM, GRU) for potentially better performance.
