import os
import collections
import numpy as np
import re
from sklearn import svm
from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB


def make_Dictionary(train_dir):

    emials = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    # print(emials,"\n")
    all_words=[]
    for mail in emials:
        with open(mail) as m:
            for i,line in enumerate(m):
                if i==2:
                    words = line.split()
                    all_words+=words
    dictionary  = collections.Counter(all_words)

    return dictionary

# dictionary = make_Dictionary("train-mails")
def remove_unimportant(dictionary):
    list_to_remove = list(dictionary.keys())
    dictionary_temp = dictionary
    for item in list_to_remove:
        if item.isalpha() == False:
            dictionary_temp.pop(item)
        elif len(item)==1:
            dictionary_temp.pop(item)
    dictionary = dictionary_temp.most_common(3000)
    dict = {}
    for x in dictionary:
        dict[x[0]] =x[1]
    return dict


def extract_features(mail_dir,dictionary):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docId = 0;
    for fil in files:
        with open(fil) as fi:
            for i,line in enumerate(fi):
                #第二行是正文
                if i==2:
                    words = line.split()
                    for word in words:
                        wordId = 0
                        for x,d in enumerate(dictionary.keys()):
                            if d == word:
                                wordId = x
                                features_matrix[docId,wordId]+=1

        docId = docId + 1
    return features_matrix

def get_label(path):
    files = [fi for fi in os.listdir(path)]

    labels = np.zeros(shape=(len(files)))
    for i,fi in enumerate(files):
        if re.match("spms",fi):
            labels[i] = 0
        else:
            labels[i] = 1
    return labels



def train_navie_bayes(X_train,y_train):
    #转成numpy 容易做切片操作
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    yi = np.zeros(2)
    #计算类别概率P(Ci)
    for x in y_train:
        if x==1:
            yi[1]+=x;
        else:
            yi[0]+=x;
    yi[0] = yi[0]/len(y_train)
    yi[1] = yi[1]/len(y_train)
    #计算P(xi|c)
    # p_ind[a][b] = P(X_b|y=a)
    p_ind = np.zeros(shape=(2,np.shape(X_train)[1]))
    for i in range(np.shape(X_train)[1]):
        for radis,x in enumerate(X_train[:,i]):
            if x==1 and y_train[radis]==0:
                p_ind[0][i]+=1;
        p_ind[0][i]/=np.shape(X_train)[0]
        p_ind[1][i] = 1-p_ind[0][i]
    return yi,p_ind


# 对单个样本进行预测，输入一个3000维的向量
def test(X_test,p_ind,yi):
    y_0_mul = 1;y_1_mul = 1
    for i,x in enumerate(X_test):
        if x!=0:
            y_1_mul*=p_ind[1][i];
        else:
            y_0_mul*=p_ind[0][i];



if __name__ == "__main__":

    # make_Dictionary("train-mails")
    #
    # dictionary = make_Dictionary("train-mails")
    # dictionary = remove_unimportant(dictionary)
    # features = extract_features("train-mails", dictionary)
    #
    #
    # for i in features[0]:
    #     print(i)
    # labels = get_label("train-mails")
    # print(features.shape,labels.shape)
    # gnb = GaussianNB()
    # gnb.fit(features,labels)
    #
    # dictionary = make_Dictionary("test-mails")
    # dictionary = remove_unimportant(dictionary)
    # test_features = extract_features("test-mails", dictionary)
    #
    # test_labels = get_label("test-mails")
    # print(test_labels)
    # pred = gnb.predict(test_features)
    #
    # print(classification_report(test_labels,pred))

    x1  = [[1,0,0],[0,1,1],[1,0,0],[0,1,1],[1,0,0],[0,1,1]]
    y1 =   [0,1,0,1,0,1]
    yi,p_ind = train_navie_bayes(x1,y1)
    print(yi)
    print(p_ind)