import unittest
from recommender_template import Recommender


class TestRecommender(unittest.TestCase):

    def __init__(self, testname):
        super(TestRecommender, self).__init__(testname)

    def setUp(self):
        """
        Set up all de variables used on this set of tests
        """
        self.rec = Recommender('data/user-item-interactions.csv',
                               'data/articles_community.csv')
        self.rec2 = Recommender('data/user-item-interactions.csv',
                                'data/df_content_cosine.csv')

        # fit recommender
        self.rec.fit()
        self.rec2.fit()

        # make recommendations
        self.expected_rank_recommendation = [
            'use deep learning for image classification',
            'insights from new york car accident reports',
            'visualize car data with brunel',
            'use xgboost, scikit-learn & ibm watson machine learning apis',
            'predicting churn with the spss random tree algorithm',
            'healthcare python streaming application demo',
            'finding optimal locations of new store using decision optimization',
            'apache spark lab, part 1: basic concepts',
            'analyze energy consumption in buildings',
            'gosales transactions for logistic regression model']
        self.expected_collaborative_recommendation = [
            'mapping points with folium',
            'a comparison of logistic regression and naive bayes ',
            'access mysql with python',
            'tidy up your jupyter notebooks with scripts',
            'analyze accident reports on amazon emr spark',
            'analyze energy consumption in buildings',
            'analyze open data sets with spark & pixiedust',
            'analyze open data sets with pandas dataframes',
            'analyze precipitation data',
            'analyzing data by using the sparkling.data library features']
        self.expected_content_recommendation = [
            'use spark r to load and analyze data',
            'machine learning for the enterprise',
            'breaking the 80/20 rule: how data catalogs transform data scientists’ productivity',
            'make machine learning a reality for your enterprise',
            'machine learning for the enterprise.',
            'this week in data science (may 23, 2017)',
            '10 powerful features on watson data platform, no coding necessary',
            'this week in data science (may 16, 2017)',
            'use the cloudant-spark connector in python notebook',
            'this week in data science (february 14, 2017)']

    def test_recommendations(self):
        """
            Test the function get_config_data
        """
        rank_recommendation = self.rec.make_recs(10, 'rank')
        collaborative_recommendation = self.rec.make_recs(
            10, 'collaborative', 10)
        content_recommendation = self.rec2.make_recs(10, 'content')
        print(collaborative_recommendation)

        self.assertEqual(
            rank_recommendation,
            self.expected_rank_recommendation)
        self.assertEqual(
            collaborative_recommendation,
            self.expected_collaborative_recommendation)
        print(content_recommendation)
        self.assertEqual(
            content_recommendation,
            self.expected_content_recommendation)
        self.assertEqual(10, len(self.expected_rank_recommendation))
        self.assertEqual(10, len(self.expected_collaborative_recommendation))
        self.assertEqual(10, len(self.expected_content_recommendation))


class TestMain():

    def execute_tests(self):
        """Execute tests
        """
        test = unittest.TestSuite()
        test.addTest(TestRecommender('test_recommendations'))

        unittest.TextTestRunner(verbosity=2).run(test)


if __name__ == '__main__':

    main = TestMain()
    main.execute_tests()
