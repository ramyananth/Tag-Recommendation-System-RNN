import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Return the top recommendations for an article
def get_recommendations(data, indices, title, cosine_sim):
    index = indices[title]

    # Compute the pairwsie similarity scores of all articles and sort them based on similarity
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 articles and return them sorted by number of claps
    sim_scores = sim_scores[1:10]
    article_indices = [i[0] for i in sim_scores]
    return data.iloc[article_indices].sort_values(by=['recommends'])

def main():
    #import os
    #os.chdir("C:\\Users\\Ramya Ananth\\Desktop\\medium")
    data = pd.read_csv('medium.csv', low_memory=False)
    input_article = 'Making the Chart That Best Illustrates My Current Music Listening Habits'

    # Create a TF-IDF vectorizer and remove stopwords and NaN
    tfidf = TfidfVectorizer(stop_words='english')
    data['text'] = data['text'].fillna('')

    # Construct TF-IDF matrix and cosine similarity matrix (for looking at text only)
    tfidf_matrix = tfidf.fit_transform(data['text'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Taking into account post tags
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(data['post_tags'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    indices = pd.Series(data.index, index=data['title']).drop_duplicates()

    text_only = get_recommendations(data, indices, input_article, cosine_sim)
    text_and_tags = get_recommendations(data, indices, input_article, cosine_sim2)

    print("Input article:", input_article)
    print("\nText similarity recommendations:")
    print(text_only['title'].values)
    print("\nText and tag similarity recommendations:")
    print(text_and_tags['title'].values)

if  __name__ =='__main__':
    main()