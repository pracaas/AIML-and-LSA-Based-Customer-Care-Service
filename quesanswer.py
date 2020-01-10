#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:48:25 2018

@author: prakash
"""

from nltk.tokenize import sent_tokenize


addrs ="/Users/prakash/Google Drive/live/Study/WRC/MSCK III sem/Projects/Project Files/FAQ.txt"


def sent_token(addrs):
    print(addrs)
    with open(addrs, 'r') as istr:
        oridata = istr.read()
        qs = sent_tokenize(oridata)
    return qs

token = sent_token(addrs)

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


def callall(qs,typ):
    ques=[]
    ans = []
    index = []
    for r in range(0,len(qs)):
        if "?" in qs[r] and "\n" not in qs[r] and "ques" in typ:
            #print(qs[r],"Question No -",r)
            ques.append(qs[r])
            index.append(r)
        else:
            #print(qs[r],"Answer No -",r)
            ans.append(qs[r])
            
    return ques,index
    '''        
    if "ques" in typ:
        return ques
    else:
        return ans
    '''

def main():
    
    ques ,index = callall(token,"ques")

    answer = []
    for i in range (0,len(ques)):
        answer.append(ans2ques(token,index[i]))
    
    #print(ques[10],"\n",answer[10])
    return ques , answer


#ques,ans = main()
#b = 90
#print(ques[b],"\n",ans[b])
