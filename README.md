# Personality-Prediction
I used online forum postings to train a Multi Layer Perceptron model to predict people's Myers Briggs Personality. I also built a web application to receive a person's text input and call on the model pipeline to predict their personality.

## Data
Data downloaded from Kaggle. The data was scraped from a personality forum, where people self-identified their personality and posted various comments and questions. 
https://www.kaggle.com/datasnaek/mbti-type

## Notebooks
Jupyter Notebook showing methods of data cleaning, processing, and engineering. I also built some visualizations and exploratory analytics to examine and evaluate the data in order to choose an appropriate model. 
After the model training, I evaluated the model with multicategorical confusion matrix and tested on new, unseen data. 
Because I used Neural Networks, we can extract the first layer of the model for word embeddings. The word vectors are then display on Google's Embedding Projector using various dimensionality reduction methods (PCA, TSNE, etc.).

[Dineise's Myers Briggs Word Embeddings](http://projector.tensorflow.org/?config=https://gist.githubusercontent.com/princessdennis/b2ce013d7991246773958202b20cea16/raw/ec338dd106cfedc15409c048fe7c956a77786a28/mbti_json)

Lastly, I pickled the data processing pipeline and the model to be used in the web application later.

## Output
Contains the pickled model, data cleaning and encoding pipeline, word embedding vectors files, and snapshot of word embedding visualization.

## Static
Python code used to clean text after they're entered in the web app.

## Templates
HTML files needed to display user interface of web app.

## predict_api.py and predict_app.py
Code to receive the texts from the web app, clean the texts, predict personality, and send the results back to the web app.
