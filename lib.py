
# coding: utf-8




from gensim import *
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import logging
import gensim
import os

from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

import matplotlib.pyplot as plt
from random import *

import pandas as pd
import numpy as np
import math

from sklearn.decomposition import PCA
import itertools


# DIRECT BIAIS

def association_Genrator(w1,w2,model,top_n=1000):
    """
    return a data frame ordered by the most similar pair of the W1,W2
    """
    # we build a dataframe with the vocabulary
    df = pd.DataFrame(list(model.wv.vocab.items()), columns=['word','count'])
    # we drop the index of the word entry
    k = df[df['word']==w1].index[0]
    j = df[df['word']==w2].index[0]
    liste = df['word'].drop([k,j]).values[:top_n]
    # we compute the length of the iteration to informethe user that the process could be long
    N=math.factorial(len(liste))/(math.factorial(len(liste)-2)*math.factorial(2))
    # init
    a = model.wv[w1]-model.wv[w2]
    S =[]
    P=[]
    i=0
    # we compute the similarity score
    for x in itertools.combinations(liste,2) :
        u = model.wv[x[0]]-model.wv[x[1]]
        S.append(cosine_similarity(a,u))
        P.append(x)
        i+=1
        # print the progression
        if i%1000000==0:
            print(i/N*100,'%')
            
    # storing and ordering
    res = pd.DataFrame(data={'pair':P,'score':S})
    res['score'] = abs(res['score']) #get absolute value
    res = res.sort_values(by='score',ascending=False,inplace=False)
    return(res)

def create_random_pair(n,model):
    """
    Function that creates n random pairs of word, using the model of Google news
    Entry: 
    - n: type:int Nunber of pair needed
    Output: 
    - Pair: type:List ; list of pair size: nx2
    """
    df = pd.DataFrame(list(model.wv.vocab.items()), columns=['word','count'])
    list_word = list(df['word'])
    shuffle(list_word)
    
    Pair=[]
    for i in range(n):
        Pair.append([list_word[i],list_word[2*i]])
    return(Pair)

def from_pair_to_subspace(Pair,model):
    """
    Entry:
    - Pair: type:List  size:nx2 ; list of pair word
    Output:
    - SubS: type:Array size: 300xn ; array of the word representation for each difference of word pair
    
    Example:
    Pair = [['man','woman']]
    Subs = np.array(model.wv(man) - model.wv(woman))
    """
    #print('There are ',len(Pair), 'pairs of words. This is also the future dimension of our subspace')
    SubS=[]
    for i in range(len(Pair)):
        SubS.append(model[Pair[i][0]]-model[Pair[i][1]])
    return(np.array(SubS))

def from_space_to_direction(SubS):
    """
    Extraction of the  principale component  of the subspace created from the differences of word pair
    Entry :
    - SubS: type:Array size: 300xn ; array of the word representation for each difference of word pair
    Output:
    - direction : principal component of the PCA
    - expl_var : explained_variance_ratio_ of the pca
    - eig_values: eig_values of the pca
    """
    n_c = int(len(SubS)/2)
    #print('the dim initial is',len(SubS[0]))
    # Now we are doinf a PCA
    pca = PCA(n_components=n_c)
    pca.fit(SubS)
    Y_pca = pca.fit_transform(SubS)
    eig_values = pca.singular_values_
    expl_var = pca.explained_variance_ratio_
    direction = pca.components_[0]
    return(direction, expl_var,eig_values)

def plot_var_eig(eig_values,expl_var):
    """
    plot the explained variance and theeigen values of the  pca
    """
    plt.figure(figsize = (20,10))
    plt.subplot(121)
    plt.bar(np.arange(len(eig_values)),eig_values)
    plt.title('The eigen values for the subspace',fontsize=18)
    
    ##
    plt.subplot(122)
    plt.bar(np.arange(len(expl_var)),expl_var)
    plt.title('The variance explained by each of the selected components',fontsize=18)
    plt.show()
    
        
def cosine_similarity(w1,w2) :
    return(np.dot(w1, w2)/(np.linalg.norm(w1)* np.linalg.norm(w2)) )

def plot_mean(n,model,rp=20) :
    Rand_var =[]
    Rand_eig  = []
    for i in range(n):
        subs_rand = from_pair_to_subspace(create_random_pair(rp,model),model)
        direction_r, expl_var_r,eig_values_r = from_space_to_direction(subs_rand)
        Rand_var.append(expl_var_r)
        Rand_eig.append(eig_values_r)
    return( plot_var_eig(np.asarray(Rand_eig).mean(axis=0) , np.asarray(Rand_var).mean(axis=0)))



# INDIRECT BIAIS


def projection(w,direction):
    """
    return the projection of w on the vector "direction"
    """
    return(np.dot(direction,np.dot(w,direction)))

def indirect_biais(w1,w2,direction):
    """
    
    """
    
    w1_t = projection(w1,direction)
    w2_t = projection(w1,direction)
    norm= np.linalg.norm(w1)* np.linalg.norm(w2)
    norm_t = np.linalg.norm(w1_t)* np.linalg.norm(w2_t)
    return((np.dot(w1, w2)-np.dot(w1_t,w2_t)/norm_t)/norm)
    

#######################
# WEAT and WEFAT      #
#######################

def s(w,A,B,model):
    """
        compute the association of the word w, with the attribute A,B
    entry:
    - w: word type: string 
    - A: attribute 1 type:list
    - B: attribute 2 type:list
    - model: Gensim W2V model
    return type:float
    """
    M=0
    N=0
    for a in A :
        M+=cosine_similarity(model[w],model[a])
    for b in B : 
        N+=cosine_similarity(model[w],model[b])
    return(M/len(A)-N/len(B))

def S(X,Y,A,B,model):
    """
       compute the difference of association for the set X and Y, with the attribute A,B
    entry:
    - X: target 1 type: list(string) 
    - Y: target 2 type: list(string)
    - A: attribute 1 type:list
    - B: attribute 2 type:list
    - model: Gensim W2V model
    return type:float
    """
    M=0
    N=0
    for x in X :
        M+=s(x,A,B,model)
    for y in Y : 
        N+=s(y,A,B,model)
    return(M-N)

def effet_size(X,Y,A,B,model) :
    """
    Compute the effect size of the association S
     entry:
    - X: target 1 type: list(string) 
    - Y: target 2 type: list(string)
    - A: attribute 1 type:list
    - B: attribute 2 type:list
    - model: Gensim W2V model
    return type:float
    """
    M=0
    N=0
    st =[]
    for x in X :
        M+=s(x,A,B,model)
    for y in Y : 
        N+=s(y,A,B,model)
    for u in np.concatenate([X,Y]):
        st.append(s(u,A,B,model))
        
    return((M/len(X)-N/len(Y)/np.std(st)))

def parties_union(X,Y):
    """
    Compute all the all the partitions of X∪Y into two sets of equal size
     entry:
    - X: target 1 type: list(string) 
    - Y: target 2 type: list(string)
    return :
    - f: list of partition type: list(list(string))
    """
    f=list(itertools.combinations(np.concatenate([X,Y]),len(X)))

    for i in range(1,len(X)):
        a= list(itertools.combinations(np.concatenate([X,Y]),i))
        f=f+a
    return(f)

def p_values(X,Y,A,B,model):
    """
    Compute the p value of the partition test S
     entry:
    - X: target 1 type: list(string) 
    - Y: target 2 type: list(string)
    - A: attribute 1 type:list
    - B: attribute 2 type:list
    - model: Gensim W2V model
    return type:float
    """
    P=[]
    a= parties_union(X,Y)
    for x,y in itertools.combinations(a,2) :
        P.append(S(x,y,A,B,model))
    n=np.size(np.where(P>S(X,Y,A,B,model)))
    print(n)
    return(n/len(P))

def wefat(w,A,B,model):
    """
    compute the WEFAT coefficient of the word w, with the attribute A,B
    entry:
    - w: word type: string 
    - A: attribute 1 type:list
    - B: attribute 2 type:list
    - model: Gensim W2V model
    return type:float
    """
    M=0
    N=0
    st=[]
    for a in A :
        M+=cosine_similarity(model[w],model[a])
    for b in B : 
        N+=cosine_similarity(model[w],model[b])
    
    for u in np.concatenate([A,B]):
        st.append(cosine_similarity(model[w],model[u]))
        
    diff_=M/len(A)-N/len(B)
    
    return(diff_/np.std(st))




