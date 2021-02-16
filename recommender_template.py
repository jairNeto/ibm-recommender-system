import pandas as pd
from recommender_functions import format_df, create_user_item_matrix, \
    get_top_articles, user_user_recs_part2, make_content_recs, tokenize


class Recommender():
    '''
    This class implements a recommender system for the best ibm articles for
    each specific user.
    At this class you can chose to user the most used techniques
    of recommendation that are rank based,
    collaborative base and content based
    '''

    def __init__(self, df_path, df_content_path):
        '''
        INPUT:
        df_path - (string) Path to a csv contaning the columns
        user_id, article_id and title
        df_content - (string) Path to a csv contaning the columns
        doc_body, doc_description, doc_full_name, doc_status and article_id

        Description:
        Init of the Recommender system
        '''
        self.df = pd.read_csv(df_path)
        self.df_content = pd.read_csv(df_content_path)
        self.df_content.drop_duplicates(subset='article_id', inplace=True)
        self.df = format_df(self.df)

    def fit(self):
        '''
        Description:
        Create the user item matrix
        '''
        self.user_item = create_user_item_matrix(self.df)

    def make_recs(self, n_top=5, rec_type='rank', user_id=None):
        '''
        INPUT:
        n_top - (int) The number of recommendations to make
        rec_type - (string) The type of the recommendation, could be:
        "rank", "collaborative" or "content".
        user_id - (int) The user_id you want make the recommendations for.

        OUTPUT:
        recs_names - (list) a list with all recommendations articles

        Description:
        Init of the Recommender system
        '''
        if rec_type == 'rank':
            return get_top_articles(n_top, self.df)
        elif rec_type == 'collaborative':
            _, recs_names = user_user_recs_part2(
                user_id, self.df, self.user_item, n_top)
            return recs_names
        else:
            _, recs_names = make_content_recs(
                self.df, self.df_content, n_top, tokenize, user_id)
            return recs_names
