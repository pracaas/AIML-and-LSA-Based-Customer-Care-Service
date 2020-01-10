


from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re, math ,operator
from collections import Counter
#from sklearn.feature_extraction.text import TfidfVectorizer
from autocorrect import spell


addrs ="/Users/prakash/Google Drive/live/Study/WRC/MSCK III sem/Projects/Project Files/FAQ.txt"


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
            print(qs[r],"Answer No -",r)
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
           
def ans2ques(qs,r): #question no- r and qs is token
    aflag = True
    ans = []
    r += 1
    while (aflag is True):
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



token = sent_token(addrs)
quess = quesans(token,"ques") 
#print(quess)
allquesno = queslist(token,"ques")
#print(allquesno)
#print(ans2ques(token,749))
#print(rtn_ques_no(token,3))
#extract(doc_complete)

def autocorrect(sent):
    t_text = word_tokenize(sent)
    t_text = [spell(word) for word in t_text]
    input_text = ' '.join(t_text)
    print(input_text)
    return input_text    


def higherlist(listname):
    dense=[]
    for  i in range (0,len(listname)):
        if listname[i] > 0.5:
            index = i
            value = listname[i]
            dense.append((index,value))
    return dense


def output(i,index,value):
    if (value > 0.5):
        print(quess[index])
        print(ans2ques(token,allquesno[index]))
        print("Max similarity ->",value, " Index = ",index,"\n")
    else:
        print(quess[index])
        print(ans2ques(token,allquesno[index]))
        print("Max similarity ->",value, " Index = ",index)
        #a= 0
    


l=True
while (l == True):
    
    
    input_text = input("> ")
    if 'bye' in input_text:
        l = False
    #input_text = autocorrect(input_text)    
    refn_input = extract(input_text)
    #print(refn_input)

    #cosincol = check_sim(token,refn_input)    
    cosincol = check_sim(quess,refn_input)
    #print(cosincol)
    j=len(sorted(i for i in cosincol if i>0.5))
    print(j , "Similar ques\n")
    #print(cosincol)
    #higherlist = higherlist(cosincol)
    #print("Higher list",higherlist)
    index, value = max(enumerate(cosincol), key=operator.itemgetter(1))
    output(i,index,value)
    #for i in range (0,len(higherlist)):
    #    output(i,higherlist[i][0],higherlist[i][1])
        
    #print("Index = ",index)
    #print(allquesno[index])
    #callall(token,"ans")
    
    #print(input_text)
    #print("Ans:",token[index], " Similarity ->",value)
    
