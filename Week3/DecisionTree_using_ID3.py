'''
Assume df is a pandas dataframe object of the dataset given
'''

from operator import indexOf
from tempfile import tempdir
import numpy as np
import pandas as pd
import math



def uni(df,attribute):
    uni=[]
    uni=df[attribute]
    unique=[]
    for x in uni:
        if x not in unique:
            unique.append(x)
    return unique
def get_avg_info_of_attribute(df,attribute):
    cou=df[attribute].nunique()
    unique=uni(df,attribute)
    lent=df.shape[0]
    count_yes=[0]*(cou)
    sum=0
    H=[0]*cou
    count=[0]*cou
    count_no=[0]*cou
    tots=0
    avg=0
    newwwww=uni(df,'play')
    for i in range(cou):
        for p in range(lent):
            j=df[attribute][p]
            z=df['play'][p]
            if (j==unique[i] and z==newwwww[0]):
                count_yes[i]=count_yes[i]+1
            elif(j==unique[i] and z==newwwww[1]):
                count_no[i]=count_no[i]+1
    for i in range(cou):
        for j in df[attribute]:
            if j==unique[i]:
                count[i]=count[i]+1
    for i in range(cou):
        sum=sum+count[i]
    for i in range(cou):
        count[i]=count[i]/sum
    for i in range(cou):
        if(count_no[i]!=0 and count_yes[i]!=0):
            n1=count_no[i]/(count_no[i]+count_yes[i])
            d1=math.log2(n1)
            n2=count_yes[i]/(count_no[i]+count_yes[i])
            d2=math.log2(n2)
            H[i]=(-1)*((n1*d1)+(n2*d2))
        else:
            H[i]=0
    for i in range(cou):
        tots=tots+count_no[i]+count_yes[i]
    for i in range(cou):
        avg=avg+(((count_yes[i]+count_no[i])/tots)*H[i])
    return avg
def helper(df,attr):
    ent=0
    sum=0
    unique=[]
    cou=df[attr].nunique()
    unique=uni(df,attr)
    count=[0]*cou
    for i in range(cou):
        for j in df[attr]:
            if j==unique[i]:
                count[i]=count[i]+1
    for i in range(cou):
            sum=sum+count[i]
    for i in range(cou):
            count[i]=count[i]/sum
    for i in range(cou):
            ent=ent+(count[i]*(math.log2(count[i])))
    return ent   
def get_entropy_of_dataset(df):
    # TODO
    ent=[]
    col=[]
    for i in df.columns:
        col.append(i)
    for i in range(len(col)):
        #print(col[i])
        sum=helper(df,col[i])
        ent.append(sum)
    entropy=max(ent)
    return (entropy*(-1))
def get_information_gain(df, attribute):
    new=get_entropy_of_dataset(df)
    avg=get_avg_info_of_attribute(df,attribute)
    return(new-avg)

def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    cols=[]
    info=[]
    dict={}
    for i in df.columns:
        cols.append(i)
    for i in range(len(cols)-1):
        ig=get_information_gain(df,cols[i])
        info.append(ig)
    for i in range(len(cols)-1):
        dict[cols[i]]=info[i]
    maxi=max(info)
    ind=info.index(maxi)
    tup=(dict,cols[ind])
    return tup
