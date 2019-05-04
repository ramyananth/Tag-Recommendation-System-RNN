Model 1:

For this model, a the Recurrent Neural Network model is implemented on Google Colab
DICE loss combined with Binary Cross Entropy (BCE) as our loss function while training the model. The steps that were followed to get the desired tags is as follows,
1. Using Snapshot for fast ensembling
2. GloVe embeddings were then applied with both GRU and LSTM combined with Max Pooling and Attention techniques
3. Trained all these tags as chunks to obtain the top tags within the dataset
4. Manual fine tuning of the punctuations was done to filter out what the training had done just to boost the scores

Model 2:
1. Extract top n tags based on boxplot to set a base for the model
2. Extract TF-IDF Feature vectors which serves as the input to the LR function
3. FastText and Word2Vec models are trained on article's text - Input to Bidirectional LSTM
4. Calculate using F1 score
