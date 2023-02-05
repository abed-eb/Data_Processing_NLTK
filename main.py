import re
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import numpy as np
from nltk.corpus import stopwords
from operator import itemgetter
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import itertools
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from yellowbrick.text import FreqDistVisualizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
ps = PorterStemmer()
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


def merge_data():
    df1 = pd.read_csv("data/movie_info.csv")
    df2 = pd.read_json("data/movie_synopsis.json", orient='index')
    df1.rename(columns={'locale_id': 'local_id'}, inplace=True)
    df_merged = pd.merge(df1, df2, on=['local_id'])
    return df_merged


def pre_process(df):
    df['clean_plot_synopsis'] = df['plot_synopsis'].apply(
        lambda x: remove_stopwords(lemmatize_words(remove_stopwords(remove_punctuations(x)))))
    print("plain text: ", df.iloc[2]['plot_synopsis'])
    print("preprocessed text: ", df.iloc[2]['clean_plot_synopsis'])
    return df


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    text = text.replace(',', ' ')
    text = text.replace('.', " ")
    # remove everything except alphabets
    text = re.sub("[^a-zA-Z]", " ", text)
    # remove whitespaces
    text = ' '.join(list(map(str.strip, text.split())))
    text = text.lower()
    return text


def remove_stopwords(text):
    pronouns = ['i', 'he', 'she', 'it', 'him', 'they', 'we', 'us', 'them']
    others = ['tell', 'go', 'take', 'man', 'one', 'back', 'try', 'two', 'time', 's', 'l', 'also', 'become', 'away',
              'next',
              'name', 'call', 'first', 'however', 'john', 'still', 'jack', 'would', 'get', 'see', 'make', 'ask', 'come',
              'day', 'new', 'know', 'later', 'want', 'look', 'help', 'year', 'another', 'like', 'well', 'harry', 'long',
              'visit', 'find', 'leave', 'ed', 'jim', 'le', 'say', 'give', 'mr', 'did', 'take', 'turn', 'end', 'film',
              'set', 'three', 'even', 'several', 'place', 'meanwhile', 'finally', 'soon', 'door', 'talk', 'use',
              'frank',
              'tom', 'continue', 'last', 'together', 'never', 'dr', 'able', 'agree', 'allow', 'begin', 'believe',
              'bring',
              'change', 'belasco', 'muffy', 'leeloo', 'mccabe', 'sub', 'kinjanja', 'hyp', 'chizhov', 'ande',
              'littlefoot',
              'simba', 'boffano', 'helsing', 'renfield', 'nosferatu', 'lestat', 'hutter', 'voysey', 'kotov',
              'pittsburgh',
              'arizona', 'winger', 'bilko', 'robbin', 'hatchett', 'mowgli', 'ripley', 'spock', 'mccoy', 'picard',
              'mcclane',
              'kelso', 'leary', 'jeffrie', 'dc', 'chekhovs', 'miss', 'jos', 'desdemona', 'brighton', 'crockett',
              'simms',
              'deckard', 'minton', 'las', 'vegas', 'dumbo', 'norbu', 'albrecht', 'roseman', 'orlok', 'katherine',
              'indy',
              'mrs', 'gotham', 'barne']

    stop_words = stopwords.words('english') + pronouns + others
    text_with_no_stop_words = [w for w in text.split() if not w in stop_words]
    return ' '.join(text_with_no_stop_words)


def lemmatize_words(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    return ' '.join(words)


def calculate_genres_count(df):
    genres_list = []
    for i in df['genre_imdb']:
        genres_list.append(str(i).split('|'))
    genres = sum(genres_list, [])
    len(set(genres))
    genres = nltk.FreqDist(genres)
    genres_count_df = pd.DataFrame({'Genre': list(genres.keys()), 'Count': list(genres.values())})
    g = genres_count_df.nlargest(columns="Count", n=50)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax = sns.barplot(data=g, x="Count", y="Genre")
    plt.show()
    return genres


def frequent_word_per_genre(df, genre):
    plot = df.loc[df['genre_imdb'].str.contains(genre, na=False), ['clean_plot_synopsis']]
    plotlist = [x for x in plot['clean_plot_synopsis'].str.split()]
    plotlist = list(itertools.chain(*plotlist))
    count = CountVectorizer()
    docs = count.fit_transform(plotlist)
    features = count.get_feature_names_out()
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle(genre, size=20)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=20)
    visualizer = FreqDistVisualizer(features=features, n=10)
    visualizer.fit(docs)
    visualizer.show()


def tfidf(df):
    tfidf_vect = TfidfVectorizer(max_df=0.7, min_df=5, max_features=None, ngram_range=(1, 3))
    plots = df['clean_plot_synopsis'].map(str)
    tf = tfidf_vect.fit_transform(plots)
    words = tfidf_vect.get_feature_names_out()
    words_tfidf_sums = tf.sum(axis=0)
    words_tfidf_scores = []
    output = []
    for i in range(len(words)):
        words_tfidf_scores.append([words[i], words_tfidf_sums[0, i]])
    words_tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    print("words with high tfidf:")
    for i in range(100):
        print('{:<15s}{:>8.2f}'.format(words_tfidf_scores[i][0],
                                       words_tfidf_scores[i][1]))
        output.append([words_tfidf_scores[i][0], words_tfidf_scores[i][1]])
        np.savetxt("tfidf_result_scores.csv", output, delimiter=",", fmt='%s')
    return words_tfidf_scores


def elbow_k_means(df):
    wcss = []
    for i in range(1, 11):
        clustering = KMeans(n_clusters=i, init='k-means++', random_state=42)
        clustering.fit(df)
        wcss.append(clustering.inertia_)

    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sns.lineplot(x=ks, y=wcss)
    plt.show()


def clustering(df, genres):
    titles = df['title'].tolist()
    synopses = df['clean_plot_synopsis'].tolist()
    # genres = [genres.append(i.split('|')) for i in df['genre_imdb']]
    # define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)  # fit the vectorizer to synopses
    dist = 1 - cosine_similarity(tfidf_matrix)

    # elbow for finding k
    elbow_k_means(tfidf_matrix)

    # KMeans clustering
    num_clusters = 8
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()

    # Words in the cluster
    print("words per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf_vectorizer.get_feature_names_out()
    for i in range(num_clusters):
        print("Cluster %d:" % i)
        for ind in order_centroids[i, :5]:
            print(' %s' % terms[ind], end=',')
        print()


def main():
    df = merge_data()
    df2 = pre_process(df)
    df2.dropna()
    genres_list = calculate_genres_count(df2)
    # for g in genres_list:
    #     frequent_word_per_genre(df2, g)
    tfidf_scores = tfidf(df2)
    clustering(df2, genres_list)

if __name__ == '__main__':
    main()
