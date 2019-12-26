import requests
from bs4 import BeautifulSoup
import pprint
from sklearn.datasets import fetch_20newsgroups 
from nltk.stem import WordNetLemmatizer 
import gensim

class Article():
    def __init__(self, url ='', title='',):
        self.url = url
        self.title = title
        self.body = ''
        self.update_date = ''
        self.extract_article_info()
        
    def extract_article_info(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text,'html.parser')
        self.body = soup.find(class_='post-body').text


class Blog():
    def __init__(self,name,url):
        self.name = name
        self.url = url
        self.articles = list()
        self.extract_articles()
    def extract_articles(self):
        titles = list()
        # Get title and link
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for title in soup.select('div.post-header > h2 > a '):
            self.articles.append(Article(title=title.text,url=title['href']))

class LDA_Model():
    def __init__(self):
        self.newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
        print("Data: %s" % self.newsgroups_train.data)
        self.newsgroups_test = fetch_20newsgroups(subset='test', shuffle = True)
        self.training_docs = self.process_training_docs(self.newsgroups_train.data)
        self.dictionary = self.create_dictionnary()
        self.model = self.create_lda_model()

    def lemmatize_stemming(self, text):
        return WordNetLemmatizer().lemmatize(text, pos='v')

    # Tokenize and lemmatize
    def preprocess(self,text):
        tokens=[]
        for token in gensim.utils.simple_preprocess(text) :
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                tokens.append(self.lemmatize_stemming(token))
        return tokens

    def process_training_docs(self,data):
        processed_docs = []        
        for doc in data:
            processed_docs.append(self.preprocess(doc))
        return processed_docs

    def create_dictionnary(self):
        dictionary = gensim.corpora.Dictionary(self.training_docs)
        dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
        return dictionary

    def create_lda_model(self):

        bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.training_docs]
        lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                    num_topics = 8, 
                                    id2word = self.dictionary,                                    
                                    passes = 10,
                                    workers = 2)
        return lda_model

    def get_main_topic(self,text_input):
        # Data preprocessing step for the unseen document
        bow_vector = self.dictionary.doc2bow(self.preprocess(text_input))
        for index, score in sorted(self.model[bow_vector], key=lambda tup: -1*tup[1]):
            print("Score: {}\t Topic: {}".format(score, self.model.print_topic(index, 5)))

if __name__ == '__main__':
    print('Creating LDA Model:')
    lda_model = LDA_Model()
    print('LDA Model successfully built !\n')

    print('initiating blog:')
    blog = Blog('Scrapping Hub' , 'https://blog.scrapinghub.com/')
    print('Blog %s successfully built !\n' % blog.name)

    for article in blog.articles:
        print('\nArticle: %s' % article.title)
        lda_model.get_main_topic(article.body)