
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

def normalized(x):
    if np.linalg.norm(x)==0:
        x_norm = x
    else :
        x_norm =x/np.linalg.norm(x)
    return(x_norm)
    

def create_random_pair(n,model):
    """
    Function that creates n random pairs of word, using the model of Google news
    Entry: 
    - n: type:int Nunber of pair needed
    Output: 
    - Pair: type:List ; list of pair size: nx2
    """
    df = pd.DataFrame(list(model.vocab.items()), columns=['word','count'])
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

def from_pair_to_direction0(Pair,model):
    """
    Extraction of the  principale component  of the subspace created from the differences of word pair with
    with the vector b-a
     Entry:
    - Pair: type:List  size:nx2 ; list of pair word
    - model: type:GensimModel
    Output:
    - direction : principal component of the PCA
    - expl_var : explained_variance_ratio_ of the pca
    - eig_values: eig_values of the pca
    """
    SubS = []
    for a, b in Pair:
        SubS.append(model[a] - model[b] )
    SubS= np.array(SubS)
    n_c =int(len(Pair)/2 )
    #print('the dim initial is',len(SubS[0]))
    # Now we are doinf a PCA
    pca = PCA(n_components=n_c)
    pca.fit(SubS)
    Y_pca = pca.fit_transform(SubS)
    eig_values = pca.singular_values_
    expl_var = pca.explained_variance_ratio_
    direction = pca.components_[0]
    return(direction, expl_var,eig_values)


def from_pair_to_direction1(Pair,model):
    """
    Extraction of the  principale component  of the subspace created from the differences of word pair with
    with the vector b-a,a-b
     Entry:
    - Pair: type:List  size:nx2 ; list of pair word
    - model: type:GensimModel
    Output:
    - direction : principal component of the PCA
    - expl_var : explained_variance_ratio_ of the pca
    - eig_values: eig_values of the pca
    """
    SubS = []
    for a, b in Pair:
        SubS.append(model[a] - model[b] )
        SubS.append(model[b] - model[a] )
    SubS= np.array(SubS)
    n_c =len(Pair) 
    #print('the dim initial is',len(SubS[0]))
    # Now we are doinf a PCA
    pca = PCA(n_components=n_c)
    pca.fit(SubS)
    Y_pca = pca.fit_transform(SubS)
    eig_values = pca.singular_values_
    expl_var = pca.explained_variance_ratio_
    direction = pca.components_[0]
    return(direction, expl_var,eig_values)



def from_pair_to_direction2(Pair, model):
    """
    Extraction of the  principale component  of the subspace created from the differences of word pair with
    with the vector b-center,a-center
     Entry:
    - Pair: type:List  size:nx2 ; list of pair word
    - model: type:GensimModel
    Output:
    - direction : principal component of the PCA
    - expl_var : explained_variance_ratio_ of the pca
    - eig_values: eig_values of the pca
    """
    num_components = len(Pair)
    matrix = []
    for a, b in Pair:
        center = (model[a] + model[b])/2
        matrix.append(model[a] - center)
        matrix.append(model[b] - center)
    matrix = np.array(matrix)
    pca = PCA(n_components = num_components)
    pca.fit(matrix)
    # bar(range(num_components), pca.explained_variance_ratio_)
    eig_values = pca.singular_values_
    expl_var = pca.explained_variance_ratio_
    direction = pca.components_[0]
    return(direction, expl_var,eig_values)


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

def Direct_Biais(corpus, Pair, model):
    """
    Return the direct biais of the corpus base on a pair of word to build a direction and a model
    entry:
    - corpus; type:list of word
    - Pair;  type:list of Pair of Word
    - model; type:GensimModel
    output:
    - db; type:float
    
    """
    direction_g, expl_var_g,eig_values_g = from_pair_to_direction2(Pair, model)
    B=0
    for word in corpus :
        B = B + abs(cosine_similarity(direction_g,model[word]))
    return(B/len(corpus))


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

##############################################################
# Infer Gender#
###############
# df = pd.read_excel(r'DATA\FR\Lexique382.xlsx',encoding='latin1')
def merge_model_lexique(lexique,model):
    """
    create a dataframe with word present in the lexiue and in our model
    """

    #Isolate the gender and create
    df_genre =  lexique[['1_ortho','5_genre']]
    df_genre = pd.DataFrame(data = df_genre.values,columns=['Mot','genre'])
    #Load the word of the model
    df_model = pd.DataFrame(list(model.vocab.items()), columns=['Mot','count'])
    # Merge the vocab of the Lexique and the model
    result = pd.merge(df_model,df_genre, on='Mot')
    #keep only the word where there is a gender
    result = result.dropna()
    
    return(result)

def genre(x):
    """
    transform the gender f,m into 0,1
    """
    if 'f' in str(x):
        return(0)
    if 'm' in str(x) :
        return(1)

def built_X_y(result,model):
    X=[]
    for word in list(result['Mot']):
        X.append(model[word])
        
    y = result['genre'].apply(genre)
    return(np.asarray(X),y)

##########################################################################
# Autre #

def load_embeddings(embeddings_path):
    with codecs.open(embeddings_path + 'words.txt', 'r', 'utf8') as f_in:
        index2word = [line.strip() for line in f_in]
     
    word2index = {w: i for i, w in enumerate(index2word)}
    wv = np.load(embeddings_path + 'matrix.npy')
        
    return wv, index2word, word2index

def buil_model(wv,index2word):
    data = pd.DataFrame(wv,
                    index=index2word[1:])
    np.savetxt('C:/Users/Robin/Documents/IMPACT/Notebooks/DATA/FR/ppmi_svd_fr/matrix.txt', data.reset_index().values, 
           delimiter=" ", 
           header="{} {}".format(len(data), len(data.columns)),
           comments="",encoding='latin1',
           fmt=["%s"] + ["%.18e"]*len(data.columns))
    model  = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Robin/Documents/IMPACT/Notebooks/DATA/FR/ppmi_svd_fr/matrix.txt',binary=False,encoding='latin1')
    return(model)

######
# contribution #
def from_pair_to_contribution(Pair, model):
    """
    Extraction of the contribution of each component for a pair of word in PCA
     Entry:
    - Pair: type:List  size:nx2 ; list of pair word
    - model: type:GensimModel
    Output:
    - direction : principal component of the PCA
    - expl_var : explained_variance_ratio_ of the pca
    - eig_values: eig_values of the pca
    """
    num_components = len(Pair)
    matrix = []
    label = []
    for a, b in Pair:
        center = (model[a] + model[b])/2
        matrix.append(model[a] - center)
        matrix.append(model[b] - center)
        label.append(str(a+'-'+b))
        label.append(str(b+'-'+a))
    
    matrix = np.array(matrix)
    pca = PCA(n_components = num_components)
    coord = pca.fit_transform(matrix)
    eigval = pca.singular_values_
    ctr = coord**2 
    contrib= pd.DataFrame(index=label)
    for j in range(num_components): 
        ctr[:,j] = ctr[:,j]/(eigval[j]) 
        contrib['CTR_'+str(j+1)] = ctr[:,j]
    return(contrib)

