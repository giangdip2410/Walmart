import pandas as pd
import numpy as np
from scipy.sparse.construct import random
from category_encoders import OrdinalEncoder
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from pandarallel import pandarallel
import re
from sklearn.decomposition import NMF, PCA, TruncatedSVD
import pickle5
from sklearn.model_selection import StratifiedKFold
from category_encoders.target_encoder import TargetEncoder
from datetime import datetime as dt
from umap import UMAP
from itertools import combinations
from sklearn.preprocessing import StandardScaler
pandarallel.initialize()
tqdm.pandas()

from contextlib import contextmanager
from time import time

@contextmanager
def timer(logger=None,format_str='{:.3f}[s]',prefix=None,suffix=None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time()
    yield
    d = time()-start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)
        
from tqdm import tqdm

def get_function(block,is_train):
    s = mapping ={
        True:'fit',
        False:'transform'
    }.get(is_train)
    return getattr(block,s)

def to_feature(input_df,blocks,is_train=False):
    out_df = pd.DataFrame()
    
    for block in tqdm(blocks,total=len(blocks)):
        func = get_function(block,is_train)
        
        with timer(prefix='create' + str(block) + ' '):
            _df = func(input_df)
        
        assert len(_df) == len(input_df),func._name_
        out_df = pd.concat([out_df,_df],axis=1)
    return out_df

class BaseBlock(object):
    def fit(self,input_df,y=None):
        return self.transform(input_df)
    
    def transform(self,input_df):
        raise NotImplementedError()

class WrapperBlock(BaseBlock):
    def __init__(self,function):
        self.function=function

    def transform(self,input_df):
        return self.function(input_df)
    
class GroupBlock(BaseBlock):        
    def __init__(self,target_col,key,func):
        self.target_col = target_col
        self.key = key
        self.func = func
        self.meta = None

    def fit(self,input_df):
        self.meta = input_df.groupby(self.key)[self.target_col].agg(self.func)
        return self.transform(input_df)
             
    def transform(self, input_df):
        return pd.merge(input_df[self.key],self.meta,how='left',on=self.key).drop(columns=self.key)\
            .add_suffix(f'_{self.key}_{self.func}')
            
class GroupshiftBlock(BaseBlock):        
    def __init__(self,key,i):
        self.key = key
        self.meta = None
        self.i = i

    def fit(self,input_df):
        self.meta = input_df.groupby(self.key)['Weekly_Sales'].agg('mean').shift(self.i).reset_index()
        return self.transform(input_df)
             
    def transform(self, input_df):
        return pd.merge(input_df[self.key],self.meta,how='left',on=self.key).drop(columns=self.key)\
            .add_prefix(f'Shift{self.i}_{self.key}_')
            
class GroupdiffBlock(BaseBlock):        
    def __init__(self,key,i):
        self.key = key
        self.meta = None
        self.i = i

    def fit(self,input_df):
        self.meta = input_df.groupby(self.key)['Weekly_Sales'].agg('mean').diff(self.i).reset_index()
        return self.transform(input_df)
             
    def transform(self, input_df):
        return pd.merge(input_df[self.key],self.meta,how='left',on=self.key).drop(columns=self.key)\
            .add_prefix(f'Diff{self.i}_{self.key}_')
            
class GrouppctBlock(BaseBlock):        
    def __init__(self,key,i):
        self.key = key
        self.meta = None
        self.i = i

    def fit(self,input_df):
        self.meta = input_df.groupby(self.key)['Weekly_Sales'].agg('mean').pct_change(self.i).reset_index()
        return self.transform(input_df)
             
    def transform(self, input_df):
        return pd.merge(input_df[self.key],self.meta,how='left',on=self.key).drop(columns=self.key)\
            .add_prefix(f'Pct{self.i}_{self.key}_')
            
class GrouprollingBlock(BaseBlock):        
    def __init__(self,key,i):
        self.key = key
        self.meta = None
        self.i = i

    def fit(self,input_df):
        self.meta = input_df.groupby(self.key)['Weekly_Sales'].agg('mean').rolling(self.i).mean().reset_index()
        return self.transform(input_df)
             
    def transform(self, input_df):
        return pd.merge(input_df[self.key],self.meta,how='left',on=self.key).drop(columns=self.key)\
            .add_prefix(f'Rolling{self.i}_{self.key}_')

class BeforesalesBlock(BaseBlock):
    def __init__(self):
        self.meta = None
        
    def fit(self,input_df):
        _meta_df = input_df[['Weekly_Sales','Year','Store','Dept','Week']].copy()
        _meta_df['Year'] += 1
        _meta_df.rename(columns={'Weekly_Sales':'Before_Sales'},inplace=True)
        self.meta = _meta_df
        return self.transform(input_df)
    
    def transform(self, input_df):
        output_df = pd.merge(input_df[['Year','Dept','Store','Week']],self.meta,how='left',on=['Year','Dept','Store','Week'])
        return output_df['Before_Sales']
    
class LabelEncodingBlock(BaseBlock):
    def __init__(self,cols:list):
        self.cols = cols
        self.oe = None

    def fit(self,input_df):
        input_df[self.cols].fillna('NAN',inplace=True)
        oe = OrdinalEncoder(cols=self.cols,handle_unknown='inpute')
        oe.fit(input_df[self.cols])
        self.oe = oe
        return self.transform(input_df)

    def transform(self, input_df):
        input_df[self.cols].fillna('NAN',inplace=True)
        return self.oe.transform(input_df[self.cols]) 