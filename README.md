Model 1:
1. Using Snapshot for fast ensembling
2. GloVe embeddings were then applied with both GRU and LSTM combined with Max Pooling and Attention techniques
3. DICE loss combined with Binary Cross Entropy - Loss function
3. Trained all these tags as chunks to obtain the top tags within the dataset
4. Manual fine tuning of the punctuations was done to filter out what the training had done just to boost the scores

Model 2:
1. Extract top n tags based on boxplot to set a base for the model
2. Extract TF-IDF Feature vectors which serves as the input to the LR function
3. FastText and Word2Vec models are trained on article's text - Input to Bidirectional LSTM
4. Calculate using F1 score

Model 3:
1. Apply KNN algorithm 
