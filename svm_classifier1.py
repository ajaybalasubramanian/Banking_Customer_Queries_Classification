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
import nltk
import numpy as np
from time import time
import pprint, json
import pickle
import nltk
import os
import sys
i = 0

'''
This class will train and test the data and will give suspected class as result
'''
class ArticleClassification(object):
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
            self.hasher = FeatureHasher(input_type='string',non_negative=True)
            self.clf =  SVC(probability=True,C=5., gamma=0.001)
            
            data_folder = self.root_dir + "/training_data"
            train_dataset = load_files(data_folder)
            
            print("Time taken to load the data=>", time()-start_time)            
            cache = dict(train=train_dataset)
            self.data_train = self.fetch_data(cache, subset='train')
            try:
                # print data_train.target_names
                print "Loading pickle"
                self.clf = pickle.load(open("model.pickle", "rb" ) )
                print "trained and ready ..."
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
                self.clf.fit(X_train, y_train)
                readselfclf = open('model.pickle', 'wb')
                pickle.dump(self.clf, readselfclf)
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
            pred = self.clf.predict(X_test)
            #print("pred=>", pred)
            self.categories = self.data_train.target_names
            index = 1
            predict_prob = self.clf.predict_proba(X_test)
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
	print result	
	print "process ends here"    
    def readAndCall(self):
        import pdb
        # pdb.set_trace()
        t1 = time()
        start_time = time()
        data_file=open('June30_data.txt')
        input_data=data_file.read().split("\n")
        total = 0
        sus_count = 0
        sus_count_clone = 0
        sus_count_dist = 0
        sus_count_dup = 0
        unsus_count = 0
        processed_title = []
        print "Started reading mongo data"
        for i in range(0,(len(input_data)-1)):
            print i
            result = {}
            mydata=input_data[i].split("\t(")
            result['title'] =mydata[0]
            result['description']=mydata[1]
            # completed_job_request = self.gm_client.submit_job('test_svm_rumour_classifier', json.dumps( result )  ,wait_until_complete=True)
            rec_result = self.testClassifier(json.dumps( result ))
            # print rec_result
            rec_result = json.loads(rec_result)
            total += 1
                        
            if rec_result['result']['suspected'] > rec_result['result']['unsuspected']:
                print "suspected :", sus_count
                sus_count += 1
                dup_flag = 0
                for title in processed_title:
                    if fuzz.ratio(rec_result['title'], title)> 80:
                       sus_count_dup += 1
                       dup_flag = 1
                       print "duplicate : ", rec_result['title']
                       break
                    # if dup_flag == 0:
                    #     sus_count_dist += 1
                processed_title.append(rec_result['title'])
                
            else:
                unsus_count +=1
        print "Processed title : ", processed_title
        print "Result for June 30"
        print "Total :", total
        print "Suspected : ", sus_count
        print "duplicates : ", sus_count_dup
        print "Unsuspected : ", unsus_count

    def testSingleRecord(self):
        while True:
            result = {}
            print "\n Enter description"
            desciption = raw_input()
	    result['complain']= desciption
            rec_result = self.testClassifier(json.dumps( result ))

if __name__ == '__main__':
    ArticleClassification().testSingleRecord()
