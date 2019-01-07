from sklearn.utils import check_random_state
from sklearn.datasets import load_files
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.externals import joblib
from sklearn.feature_extraction.text import FeatureHasher
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
from svm_classifier_issues import IssueClassification
import nltk
import numpy as np
from time import time
import pprint, json
import pickle, pdb
import nltk
import os
import sys
i = 0

'''
This class will train and test the data and will give suspected class as result
'''
class ProductClassification(object):
    """
    Init for complain classification
    """
    def __init__(self):
        # self.gm_worker = gearman.GearmanWorker(['localhost:4730'])
        # self.gm_worker.register_task('test_svm_rumour_classifier', self.testClassifier)
        self.root_dir = os.getcwd()
        self.trainClassifier()
        
    """
    Function to fetch the data from cache
    @cache  <dict>  consist of training data
    """
    def fetch_data(self, cache, data_home=None, subset='train', categories=None,
                       shuffle=True, random_state=42):
        if subset in ('train', 'test'):
            data = cache[subset]
        else:
            raise ValueError(
                "subset can only be 'train', 'test' or 'all', got '%s'" % subset) 
        if shuffle:
            random_state = check_random_state(random_state)
            indices = np.arange(data.target.shape[0])
            random_state.shuffle(indices)
            data.filenames = data.filenames[indices]
            data.target = data.target[indices]
            # Use an object array to shuffle: avoids memory copy
            data_lst = np.array(data.data, dtype=object)
            data_lst = data_lst[indices]
            data.data = data_lst.tolist()
        return data
    
    """
    For custom tokenizing the text, removed stop words from text
    @text   <type 'str'>    text which needs to get tokenized
    @return <type 'str'>    tokens
    """
    def token_ques(self, text):
        things_to_replace = ['?']
        things_to_replace += stopwords.words('english')
        #wh_word = None
        for tok in text.split('\n'):
            original_query = tok
            # 1. Stemming
            # 2. POS consideration verb, adjectives
            query_pos_tags = nltk.pos_tag(word_tokenize(tok))     
            for word in things_to_replace:
                tok = tok.lower()
                tok = tok.strip("  ")
            for word in word_tokenize(tok):
                yield word.lower()
    
    """
    Train classifier
    """
    def trainClassifier(self):
        try:
            t1 = time()
            start_time = time()
            self.hasher = FeatureHasher(input_type='string')
            self.clf =  SVC(probability=True,C=5., gamma=0.001)
            
            data_folder = self.root_dir + "/training_data_product"
            train_dataset = load_files(data_folder)
            
            print("Time taken to load the data=>", time()-start_time)            
            cache = dict(train=train_dataset)
            self.data_train = self.fetch_data(cache, subset='train')
            try:
                self.clf_product = pickle.load(open("model_product.pickle", "rb" ) )
            except:
                import traceback
                print traceback.format_exc()
                print "Generating pickles"
                training_data = []
                for text in self.data_train.data:
                    text = text.decode('utf-8','ignore')
                    training_data.append(text)
                raw_X = (self.token_ques(text) for text in training_data)  #Type of raw_X  <type 'generator'>
                X_train = self.hasher.fit_transform(raw_X)
                y_train = self.data_train.target      
                self.clf_product.fit(X_train, y_train)
                readselfclf = open('model.pickle', 'wb')
                pickle.dump(self.clf_product, readselfclf)
                readselfclf.close()
                print "Training ended"
                print("Classifier trained ...")
                print("time taken=>", time()-t1)
                
        except Exception:
            import traceback
            print traceback.format_exc()
            
    """
    Function to test classifier
    """
    def testClassifier(self, record):
        try:
            query = json.loads(record)
            # return json.dumps(lookup_result)
            query = query['complain']
            result = {}
            test_data = [query]
            raw_X = (self.token_ques(text) for text in test_data)
            X_test = self.hasher.fit_transform(raw_X)
            pred = self.clf_product.predict(X_test)
            #print("pred=>", pred)
            self.categories = self.data_train.target_names
            index = 1
            predict_prob = self.clf_product.predict_proba(X_test)
            for doc, category_list in zip(test_data, predict_prob):
                # print('\n\n')
                category_list = sorted(enumerate(category_list), key=lambda x:x[1], reverse=True)
                i = 0
                for val in category_list:
                    #print('%r => %s => %0.2f' % (doc, self.categories[val[0]], (float(val[1]) * 100)))
                    result[self.categories[val[0]]] = val[1] * 100
        except Exception:
            import traceback
            print traceback.format_exc()
        return result
    
    def testSingleRecord(self):
        while True:
            query = {}
            print "\n Enter Complain text"
            desciption = raw_input()
            query['complain']= desciption
            result_product = self.testClassifier(json.dumps( query ))
	    result_product_type  = max(result_product, key=result_product.get)
	    print "\nClassified Product name with score :"
	    print str(result_product_type) + " - " + str(result_product[result_product_type])
            result_issue = IssueClassification().testClassifier(json.dumps( query ))
	    result_issue_type  = max(result_issue, key=result_issue.get)
	    print "\nClassified Issue name with score :"
	    print str(result_issue_type) + " - " + str(result_issue[result_issue_type])
if __name__ == '__main__':
    ProductClassification().testSingleRecord()
