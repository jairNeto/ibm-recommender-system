import nltk
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from numpy.linalg import norm
import numpy as np
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


def email_mapper(df):
    '''
    INPUT:
    df - (pandas dataframe) df as defined at the top of the notebook
    OUTPUT:
        email_encoded - (list) A list with the user ids
    Description:
        Map the email to a user_id
    '''
    coded_dict = dict()
    cter = 1
    email_encoded = []

    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter += 1

        email_encoded.append(coded_dict[val])
    return email_encoded


def format_df(df):
    email_encoded = email_mapper(df)
    del df['email']
    df['user_id'] = email_encoded
    return df


def get_top_articles(n, df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles

    '''
    # Your code here
    top_articles_id = df.groupby('article_id').count(
    ).sort_values('user_id', ascending=False).index[:n]
    top_articles = []
    for id in top_articles_id:
        top_articles.append(df[df['article_id'] == id].iloc[0].title)

    # Return the top article titles from df (not df_content)
    return top_articles


def get_top_article_ids(n, df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles

    '''
    # Your code here
    top_articles = list(df.groupby('article_id').count(
    ).sort_values('user_id', ascending=False).index[:n])

    return top_articles  # Return the top article ids


def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns

    OUTPUT:
    user_item - user item matrix

    Description:
    Return a matrix with user ids as rows and article ids on the columns
    with 1 values where a user interacted with
    an article and a 0 otherwise
    '''
    # Fill in the function here
    user_item = df.groupby(['user_id', 'article_id'])['title'].apply(
        lambda x: 1 if len(x) > 0 else 0).unstack()
    user_item.fillna(0, inplace=True)
    return user_item  # return the user_item matrix


def find_similar_users(user_id, user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise

    OUTPUT:
    similar_users - (list) an ordered list where the closest users
                    (largest dot product users)
                    are listed first

    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered

    '''
    # compute similarity of each user to the provided user
    similar_mat = user_item.dot(user_item.loc[user_id].T)
    # sort by similarity
    most_similar_users = \
        similar_mat.sort_values(ascending=False).index.tolist()

    most_similar_users.remove(user_id)

    return most_similar_users


def get_article_names(article_ids, df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    article_names - (list) a list of article names associated with the
                    list of article ids
                    (this is identified by the title column)
    '''
    # Your code here
    article_names = []
    for a_id in article_ids:
        df_filtered = df[df['article_id'] == float(a_id)]
        if len(df_filtered) > 0:
            article_names.append(df_filtered.iloc[0].title)
        else:
            article_names.append('')

    # Return the article names associated with list of article ids
    return article_names


def get_user_articles(user_id, user_item, df):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise
    df - (pandas dataframe) dataframe with the column article_id and title

    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the
                    list of article ids
                    (this is identified by the doc_full_name
                     column in df_content)

    Description:
    Provides a list of the article_ids and article titles that
    have been seen by a user
    '''
    # Your code here
    article_ids = list(user_item.columns[user_item.loc[float(user_id)] == 1])
    if df is not None:
        # Getting the articles with the higher number of interactions
        article_ids = \
            list(df.groupby('article_id').count().loc[article_ids].sort_values(
                'title', ascending=False).index)
    article_ids = [str(a_id) for a_id in article_ids]
    article_names = get_article_names(article_ids, df)

    return article_ids, article_names  # return the ids and names


def get_top_sorted_users(user_id, df, user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook
    user_item - (pandas dataframe) matrix of users by articles:
            1's when a user has interacted with an article, 0 otherwise


    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the
                    provided user_id
                    num_interactions - the number of articles viewed by
                    the user - if a u

    Other Details - sort the neighbors_df by the similarity and then by
                    number of interactions where
                    highest of each is higher in the dataframe

    '''
    # Your code here
    pandas_dict = {'neighbor_id': [], 'similarity': [], 'num_interactions': []}
    df_grouped = df.groupby('user_id').count()
    user_ids = user_item.index
    for id in user_ids:
        if id != user_id:
            similarity = user_item.loc[id] @ user_item.loc[user_id]
            pandas_dict['similarity'].append(similarity)
            pandas_dict['neighbor_id'].append(id)
            pandas_dict['num_interactions'].append(df_grouped.loc[id].title)
    neighbors_df = pd.DataFrame(pandas_dict)
    neighbors_df.sort_values(
        ['similarity', 'num_interactions'], inplace=True, ascending=False)

    return neighbors_df  # Return the dataframe specified in the doc_string


def user_user_recs_part2(user_id, df, user_item, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user

    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title

    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and
    provides them as recs
    Does this until m recommendations are found

    Notes:
    * Choose the users that have the most total article interactions
    before choosing those with fewer article interactions.

    '''
    if user_id is None:
        raise ValueError('User ID could not be None!')
    # Your code here
    neighbors_df = get_top_sorted_users(user_id, df, user_item)

    article_ids, _ = get_user_articles(user_id, user_item, df)
    article_ids = np.array(article_ids)
    recs = []
    for _, col in neighbors_df.iterrows():
        # At this function I did the sort by the articles
        # with the most total interections
        article_neigh_ids, _ = get_user_articles(
            int(col['neighbor_id']), user_item, df=df)
        diff_movies = np.setdiff1d(np.array(article_neigh_ids), article_ids)
        num_missing_movies = m - len(recs)
        if len(diff_movies) > num_missing_movies:
            recs.extend(diff_movies[:num_missing_movies])
            article_ids = \
                np.append(article_ids, diff_movies[:num_missing_movies])
        else:
            recs.extend(diff_movies)
            article_ids = np.append(article_ids, diff_movies)

        if len(recs) == m:
            break

    return recs, get_article_names(recs, df)


def tokenize(text):
    '''
    INPUT:
    text - (string) The article text

    OUTPUT:
    clean_tokens_list - (list) a list with tokens
    Description:
    Tokenize a string
    '''
    text = text.lower()
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    clean_tokens_list = []
    for tok in tokens:
        lemmatizer_tok = lemmatizer.lemmatize(tok).strip()
        clean_tok = stemmer.stem(lemmatizer_tok)
        clean_tokens_list.append(clean_tok)

    return clean_tokens_list


def generate_corpus(df, df_content):
    '''
    INPUT:
    df - (pandas dataframe) Dataframe contaning the columns
    user_id, article_id and title
    df_content - (pandas dataframe) Dataframe contaning the columns
    doc_body, doc_description, doc_full_name, doc_status and article_id

    OUTPUT:
    corpus - (list) a list with all the texts from all the rows concatenated

    Description:
    Get only the articles that the user did not view, fill the na with an
    empty string, generate a new columns that is the concatenation of
    doc_body, doc_full_name and doc_description.
    Return that new column
    '''
    df_content['doc_body'] = df_content['doc_body'].fillna('')
    df_content['doc_description'] = df_content['doc_description'].fillna('')
    df_content['joined_text'] = df_content['doc_body'] + ' ' + \
        df_content['doc_full_name'] + df_content['doc_description']

    return list(df_content['joined_text'])


def get_cosine_similarities_list(df_content):
    '''
    INPUT:
    df_content - (pandas dataframe) Dataframe contaning the columns
    doc_body, doc_description, doc_full_name, doc_status and article_id

    OUTPUT:
    cosines_simialirities_list - (list) a list with the cosine similarity
    for all articles

    Description:
    Calculate the cosine similarity between all articles
    '''

    cosines_simialirities_list = []
    for i, col in df_content.iterrows():
        cosine_sim = 0
        for j, col2 in df_content.iterrows():
            vector1 = np.array(col.vector_representation)
            vector2 = np.array(col2.vector_representation)
            cosine_sim += vector1 @ vector2/(norm(vector1)*norm(vector2))

        cosines_simialirities_list.append(cosine_sim - 1)

    return cosines_simialirities_list


def get_tfidf_list(corpus, tokenize):
    '''
    INPUT:
    corpus - (list) a list with the texts from each article
    tokenize - (function) a tokenize function

    OUTPUT:
    tfidf_arr_list - (list) a vector representation of the texts

    Description:
    Generate a vector representation of the texts
    '''

    # initialize count vectorizer object
    vect = CountVectorizer(tokenizer=tokenize)
    # get counts of each token (word) in text data
    X = vect.fit_transform(corpus)
    # initialize tf-idf transformer object
    transformer = TfidfTransformer(smooth_idf=False)
    # use counts from count vectorizer results to compute tf-idf values
    tfidf = transformer.fit_transform(X)
    # convert sparse matrix to numpy array to view
    tfidf_arr = tfidf.toarray()

    return tfidf_arr.tolist()


def make_content_recs(df, df_content, n_top, tokenize, user_id=None):
    '''
    INPUT:
    df - (pandas dataframe) Dataframe contaning the columns
    user_id, article_id and title
    df_content - (pandas dataframe) Dataframe contaning the columns
    doc_body, doc_description, doc_full_name, doc_status and article_id
    n_top - (int) number of articles to return
    tokenize - (function) a tokenize function
    user_id - (int) a user id

    OUTPUT:
    recs_id - (list) list with the recommended articles ids
    names_recs - (list) list with the recommended articles names

    Description:
    Make recommendations using NLP
    '''
    if 'vector_representation' not in df_content.columns:
        corpus = generate_corpus(df, df_content, user_id)

        df_content['vector_representation'] = get_tfidf_list(corpus, tokenize)

        cosines_simialirities_list = get_cosine_similarities_list(df_content)

        df_content['cosine_similarity'] = cosines_simialirities_list
        df_content.sort_values('cosine_similarity',
                               inplace=True, ascending=False)

    df_content_filterd_user_id = df_content[~df_content.article_id.isin(
        df[df.user_id == user_id].article_id)]
    df_content_filterd_article_id = \
        df_content_filterd_user_id[df_content_filterd_user_id.article_id.isin(
            df.article_id.unique())]
    ids_recs = list(df_content_filterd_article_id.article_id)[:n_top]

    return ids_recs, get_article_names(ids_recs, df)
