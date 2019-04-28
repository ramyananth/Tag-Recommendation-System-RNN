# Tag_Recommendation_System_Using_GloveEmbedding

For this model, a the Recurrent Neural Network model is implemented on Google Colab
DICE loss combined with Binary Cross Entropy (BCE) as our loss function while training the model. T. The steps that were followed to get the desired tags is as follows,
1. Using Snapshot for fast ensembling. This is because, while running the model on few training sets
2. GloVe embeddings were then applied with both GRU and LSTM combined with Max Pooling and Attention techniques
3. Trained all these tags as chunks to obtain the top tags within the dataset
4. Manual fine tuning of the punctuations was done to filter out what the training had done just to boost the scores
