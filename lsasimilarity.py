#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 08:19:01 2018

@author: prakash
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
#import re, math ,operator
from collections import Counter
#from sklearn.feature_extraction.text import TfidfVectorizer
from autocorrect import spell
import gethbasedata as hb

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
#from nltk.tokenize import sent_tokenize
import numpy as np


addrs ="/Users/prakash/Google Drive/live/Study/WRC/MSCK III sem/Projects/Project Files/FAQ.txt"

lsachecker = False 

def sent_token(addrs):
    with open(addrs, 'r') as istr:
        oridata = istr.read()
        qs = sent_tokenize(oridata)
    return qs


def callall(qs,typ):
    ques=[]
    ans = []
    for r in range(0,len(qs)):
        if "?" in qs[r] and "\n" not in qs[r] and "ques" in typ:
            print(qs[r],"Question No -",r)
            ques.append(r)
        else:
            print(qs[r],"Answer No -",r,"\n")
            ans.append(r)
    #if "ques" in typ:
    #    return ques
    #elif "ans" in typ:
    #    return ans
    

def queslist(qs,typ):
    ques=[]
    ans = []
    for r in range(0,len(qs)):
        if "?" in qs[r] and "\n" not in qs[r]:
            #print(qs[r],"Question No -",r)
            ques.append(r)
        else:
            #print(qs[r],"Answer No -",r)
            ans.append(r)
    if "ques" in typ:
        return ques
    elif "ans" in typ:
        return ans
    

def ansques_combo(token):
    ques_dataset = reset("ques")
    
    



def quesans(qs,typ,num=0): # if no any num passed it is for all loop . for sepecific num it returns 1 ans or 1 ques
    
    ques=[]
    ans = []
    leng = 0
    if num != 0:
        leng = num
        if "ans" in typ:
            leng += 1
        for r in range(leng,leng+1):
            if "?" in qs[r] and "\n" not in qs[r]:
                qok = qs[r]
                ques.append(qok)
            else:
                aok = qs[r]
                ans.append(aok)
    else:
        for r in range(0,len(qs)):
            if "?" in qs[r] and "\n" not in qs[r]:
                qok = qs[r]
                ques.append(qok)
            else:
                aok = qs[r]
                ans.append(aok)
        
    if "ques" in typ:
        return ques
    elif "ans" in typ:
        return ans
    
def quesans_no(qs): #return ques list and index list
    
    
    ques=[]
    for r in range(0,len(qs)):
        if "?" in qs[r] and "\n" not in qs[r]:
            qok = (qs[r],r)
            ques.append(qok)
    return ques
           
def ans2ques(qs,r): #question no- r and qs is token
    aflag = True
    ans = []
    r += 1
    while (aflag is True):
        if len(qs) == r:
            aflag = False
            ok = ' '.join(ans)
            return ok
            
        if "?" in qs[r] and "\n" not in qs[r]:
            aflag = False
            ok = ' '.join(ans)
            return ok
        else:
            aok = qs[r]
            ans.append(aok)
        r += 1

def rtn_ques_no(qs,x):
    ans=[]
    ques=[]
    for r in range(0,len(qs)):
            if "?" in qs[r] and "\n" not in qs[r]:
                qok = qs[r]
                ques.append(qok)
            else:
                aok = qs[r]
                ans.append(aok)
    return ques
    
def extract(doc_complete):
    word_list = word_tokenize(doc_complete)
    filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    #print(filtered_words)

    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    #wnl.lemmatize(word)
    ok = []
    ps = PorterStemmer()
    for word in filtered_words:
        ok.append(wnl.lemmatize(word))
    #print(wnl.lemmatize(word) if wnl.lemmatize(word).endswith('e') else ps.stem(word)))
    #print(ps.stem(word))
    test = ' '.join(ok)
    #print(test)
    return test


def cosin_sim(text1,text2):
    WORD = re.compile(r'\w+')

    def get_cosine(vec1, vec2):
         intersection = set(vec1.keys()) & set(vec2.keys())
         numerator = sum([vec1[x] * vec2[x] for x in intersection])

         sum1 = sum([vec1[x]**2 for x in vec1.keys()])
         sum2 = sum([vec2[x]**2 for x in vec2.keys()])
         denominator = math.sqrt(sum1) * math.sqrt(sum2)

         if not denominator:
            return 0.0
         else:
            return float(numerator) / denominator

    def text_to_vector(text):
         words = WORD.findall(text)
         return Counter(words)

    #text1 = 'What do I need to know about my warranty?'
    #text2 = 'What is important in my warranty?'

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

    cosine = get_cosine(vector1, vector2)

    return cosine


def check_sim(quess,inp):
    cosin = []
    for r in range (0,len(quess)):
        text1 = quess[r].lower()
        text1 = extract(text1)
        text2 = inp.lower()
        #print("Text before similarity",text1 , text2)
        #print(extract(text1))
        #print(extract(text2))
        cosin.append(cosin_sim(extract(text1),extract(text2)))
    return cosin





def autocorrect(sent):
    t_text = word_tokenize(sent)
    t_text = [spell(word) for word in t_text]
    input_text = ' '.join(t_text)
    #print(input_text)
    return input_text    


def higherlist(listname):
    #print(listname)
    dense=[]
    for  i in range (0,len(listname)):
        if listname[i] > 0.5:
            index = i
            value = listname[i]
            dense.append((index,value))
    #print(dense)
    return dense


def output(i,index,value):
    if (value > 0.5):
        #print("\n")
        #print(ques_dataset[index])
        print(ans2ques(token,allquesno[index]))
        print("\nMax similarity ->",value, " Index = ",index,"\n")
    else:
        
        #print(ques_dataset[index])
        print(ans2ques(token,allquesno[index]))
        print("Max similarity ->",value, " Index = ",index)
        #a= 0


def localprint(hrlist):
    #print(hrlist)
    for i in range (1,len(hrlist)):
    
        output(i,hrlist[i][0],hrlist[i][1],)
    

def allinone(lst):
    
    vectorizer = TfidfVectorizer(stop_words='english',use_idf=True, ngram_range=(1,1))
    X = vectorizer.fit_transform(lst)
    #print(lst)
    #print(X.toarray())
    #print(vectorizer.get_feature_names())
    #print(X)
    from sklearn.metrics.pairwise import cosine_similarity
    #print("cosine similarity\n")
    cs = cosine_similarity(X[-1], X)
    #print(cs)
    #print(cs)
    #print("It is from not LSA")
    return cs


def allinoneLSA(lst):
    #print("I am inside LSA")
    #query = input("Tell me: ")

    #example.append(query)
    #print(len(lst),": Length of List of question")
    #print(lst)
    #print(len(example))
    #vectorizer = TfidfVectorizer(min_df =2,use_idf=True, stop_words='english')
    vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
    dtm = vectorizer.fit_transform(lst)
    
    # Get words that correspond to each column
    vectorizer.get_feature_names()
    #print(len(vectorizer.get_feature_names()),": Length of features")
    lsa = TruncatedSVD(200)
    print(type(lsa),"type of lsa")
    
    dtm_lsa = lsa.fit_transform(dtm.astype(float))
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
    #print(dtm_lsa,'dtm_lsa')
    # Compute document similarity using LSA components
    similarity = np.array(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)
    print(len(similarity),": Length of similarity")
    #print(type(example))
    cs =similarity[-1]
    print("It is from LSA")
    return cs
        

def reset(ss):
    if "ques" in ss:
        ques_dataset = quesans(token,"ques")
        #print(ques_dataset)
        return ques_dataset
    

def anstheques(rowkey):
    #print(type(rowkey))
    #print(rowkey)
    answ = hb.getdata('ans',rowkey)
    #print(answ)
    return answ
#for question matching 
        
def listmaker(hrlist):
    index = []
    question = []
    answr = []
    sim = []
    
    dataset = reset_dataset('ques')
    answerindex = reset_dataset('ans')
    '''
    print("Question Data set at the time of reset\n")
    for i in range (0,len(dataset)):
        print(dataset[i],"-",i)
    print(len(dataset),"Length of dataset")
    print(hrlist,"This is hrlist")
    '''
    for i in range (0,len(hrlist)):
        sim.append(hrlist[i][1])
        indexval = hrlist[i][0]
        index.append(indexval)
        question.append(dataset[indexval])
        #print("Question in listmaker",dataset[indexval]," with indexvalue :",indexval)
        ansindex = answerindex[indexval]
        #answer.append(ans2ques(token,allquesno[indexval]))
        answr.append(anstheques(ansindex)) # In hb row key is string
    #print(list(zip(index,question,answr,sim)))
    return list(zip(index,question,answr,sim))
    #print(index , answer ,question)
    
def changemusic(input_text,ques_dataset):
    
    ques_dataset.append(input_text)
    #print(ques_dataset)
    
    ques_dataset = [i.replace("?","") for i in ques_dataset]
    #print(ques_dataset)
    #print(len(ques_dataset),"After append")
    '''
    if lsa is True:
        cs = allinoneLSA(ques_dataset)
        cssl = cs.tolist()
        
    else:
        cs = allinone(ques_dataset)
        csl = cs.tolist()
        cssl =csl[0]
    '''
    
    cs = allinone(ques_dataset)
    csl = cs.tolist()
    
    #print(csl,"It cssl after allinone")
    cssl = csl[0]
    del cssl[-1]
    hrlist = higherlist(cssl) 
    hrlist = sorted(hrlist, key=lambda x: x[1],reverse=True)

    #ques_dataset = quesans(token,"ques")
    #print(len(ques_dataset),"length of ques_dataset")
    #localprint(hrlist)
    '''
    for i in range (1,len(hrlist)):
    
        output(i,hrlist[i][0],hrlist[i][1],)
    
    ques_dataset = quesans(token,"ques")
    
    '''
    
    return listmaker(hrlist)
    


def othermusic(input_text,ques_dataset):
    
    ques_dataset.append(input_text)    
    ques_dataset = [i.replace("?","") for i in ques_dataset]    
    cs = allinoneLSA(ques_dataset)
    #print(cs)
    csl = cs.tolist()
    del csl[-1]
    hrlist = higherlist(csl) 
    hrlist = sorted(hrlist, key=lambda x: x[1],reverse=True)
    ques_dataset = quesans(token,"ques")
    #localprint(hrlist)
    
    index = []
    question = []
    answer = []
    sim = []
    #print(hrlist,"HRLIST")
    for i in range (1,len(hrlist)):
        sim.append(hrlist[i][1])
        indexval = hrlist[i][0]
        #print(indexval)
        index.append(indexval)
        
        question.append(ques_dataset[indexval])
        answer.append(ans2ques(token,allquesno[indexval]))
    return list(zip(index,question,answer,sim))
    
    

def insert2db(question,answer):
    hb.givedata(question,answer)



token = sent_token(addrs)

ques_dataset = quesans(token,"ques") 

allquesno = queslist(token,"ques")

queswithno = quesans_no(token)


def checkinputlsa(text):
    input_text = text
    input_text = input_text.replace("?","")
    gg = othermusic(input_text,reset("ques"))
    return gg

def reset_dataset(typ):
    dataset, answerindex = hb.getdata('ques')
    '''
    print("Question Data set at the time of reset\n")
    for i in range (0,len(dataset)):
        print(dataset[i],"-",i)
    print(len(dataset),"Length of dataset")
    '''
    if 'ans' in typ:
        return answerindex
    else:
        return dataset



dataset, answerindex = hb.getdata('ques')

def checkinput(text):
    input_text = text
    input_text = input_text.replace("?","")
    #ll = changemusic(input_text,reset("ques"))
    ll = changemusic(input_text,reset_dataset('ques'))
    
    #if not ll:
    #    return ""
    #else :
    return ll

def test():
    ok = checkinputlsa("where is the active eco button?")
    return ok

