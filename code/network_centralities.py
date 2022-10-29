import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os 
import pickle
# import networkx.algorithms.community as nx_comm
# https://python-louvain.readthedocs.io/en/latest/api.html
from community import community_louvain
from sklearn.preprocessing import MinMaxScaler
import datetime
#%%
city = 'seoul'
covid = 'before'


#%%

if city == 'seoul':
    if covid == 'before':
        with open('../data/seoul/따릉이_대여이력_2019v3.pkl', 'rb') as f:
            df = pickle.load(f, encoding='utf-8')
            
    elif covid == 'after':
        time_frame = pd.read_excel(f'../data/covid/{city}_frame.xlsx')
        with open('../data/seoul/따릉이_대여이력_2020v3.pkl', 'rb') as f:
            df20 = pickle.load(f, encoding='utf-8')
        with open('../data/seoul/따릉이_대여이력_2021v3.pkl', 'rb') as f:
            df21 = pickle.load(f, encoding='utf-8')
        with open('../data/seoul/따릉이_대여이력_202201-202205v3.pkl', 'rb') as f:
            df22 = pickle.load(f, encoding='utf-8')
        df = pd.concat([df20,df21,df22])
        del df20, df21, df22
        
    station = pd.read_csv('../data/seoul/대여소정보_20220803.csv')
elif city=='daejeon':
    if covid == 'before':
        with open('../data/daejeon/2019v2.pkl', 'rb') as f:
            df = pickle.load(f, encoding='utf-8')
            
    elif covid == 'after':
        time_frame = pd.read_excel(f'../data/covid/{city}_frame.xlsx')
        with open('../data/daejeon/2020v2.pkl', 'rb') as f:
            df20 = pickle.load(f, encoding='utf-8')
        with open('../data/daejeon/2021v2.pkl', 'rb') as f:
            df21 = pickle.load(f, encoding='utf-8')
        with open('../data/daejeon/2022(5월말까지)v2.pkl', 'rb') as f:
            df22 = pickle.load(f, encoding='utf-8')
        df = pd.concat([df20,df21,df22])
        del df20, df21, df22
    
    station = pd.read_csv('../data/daejeon/타슈 대여소정보_220601v2.csv')

#%% 

if covid == 'after':
    df_dict = {}
    
    for i,s,e in zip(time_frame.index,time_frame['start_date'],time_frame['end_date']):
        df_dict[i] = df[(df['rent_time'] >= s) & (df['rent_time'] < e)]
        
    del df
        
#%% calculate centrality
# https://frhyme.github.io/python-lib/network-centrality/
def return_centralities_as_dict(input_g):
    # weighted degree centrality를 딕셔너리로 리턴
    def return_weighted_degree_centrality(input_g, normalized=False):
        w_d_centrality = {n:0.0 for n in input_g.nodes()}
        for u, v, d in input_g.edges(data=True):
            w_d_centrality[u]+=d['weight']
            w_d_centrality[v]+=d['weight']
        if normalized==True:
            weighted_sum = sum(w_d_centrality.values())
            return {k:v/weighted_sum for k, v in w_d_centrality.items()}
        else:
            return w_d_centrality
    def return_closeness_centrality(input_g):
        new_g_with_distance = input_g.copy()
        for u,v,d in new_g_with_distance.edges(data=True):
            if 'distance' not in d:
                d['distance'] = 1.0/d['weight']
        return nx.closeness_centrality(new_g_with_distance, distance='distance')
    def return_betweenness_centrality(input_g):
        return nx.betweenness_centrality(input_g, weight='weight')
    def return_pagerank(input_g):
        return nx.pagerank(input_g, weight='weight')
    return {
        'weighted_deg':return_weighted_degree_centrality(input_g),
        'closeness_cent':return_closeness_centrality(input_g), 
        'betweeness_cent':return_betweenness_centrality(input_g),
        'pagerank':return_pagerank(input_g)
    }

#%%

# 겹치는 station
sta = station[(station['start_date'] <= '2019-02-19') & (station['end_date'] >= '2022-05-19')]['ID']

def make_cent_df(df):
    
    n_of_days = df['rent_time'].apply(lambda x : x.strftime('%Y-%m-%d')).nunique()
    df = df[['rent_id','return_id']]
    df = df[(df['rent_id'].isin(sta)) & (df['return_id'].isin(sta))]
    df_count = df.groupby(['rent_id','return_id']).size().reset_index()
    df_count.columns = ['rent_id','return_id','weight']
    
    df_count['weight'] = df_count['weight'] / n_of_days # 기간으로 나눔
    # print(df_count['weight'].describe())
    # df_count = df_count[df_count['weight'] >= 0.01].reset_index(drop=True) #threshold
    df_count['weight'] = df_count['weight']/max(df_count['weight'])
    
    G = nx.from_pandas_edgelist(df_count,source='rent_id', target='return_id',edge_attr='weight', create_using = nx.Graph())

    return return_centralities_as_dict(G)
#%%

if covid == 'after':
    central_dict = {}
    
    for i,d in enumerate(df_dict.values()):
        central_dict[i] = pd.DataFrame(make_cent_df(d)).sort_index()
        
    with open(f'../data/{city}/frame_dict', 'wb') as f:
            pickle.dump(central_dict, f)

elif covid == 'before':
    central_dict = pd.DataFrame(make_cent_df(df)).sort_index()
    with open(f'../data/{city}/2019_dict', 'wb') as f:
            pickle.dump(central_dict, f)
