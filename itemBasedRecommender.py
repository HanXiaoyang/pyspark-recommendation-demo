#-*- coding:utf8 -*-
# pySpark实现的基于物品的协同过滤

import sys
from collections import defaultdict
from itertools import combinations
import numpy as np
import random
import csv
import pdb

from pyspark import SparkContext

def parseVector(line):
    '''
        解析数据，key是item，后面是user和打分
    '''
    line = line.split("|")
    return line[0],(line[1],float(line[2]))

def sampleInteractions(user_id,items_with_rating,n):
    '''
        如果某个用户打分行为特别多，可以选择适当做点下采样
    '''
    if len(items_with_rating) > n:
        return user_id, random.sample(items_with_rating,n)
    else:
        return user_id, items_with_rating

def findItemPairs(user_id,items_with_rating):
    '''
        对每个用户的打分item，组对
    '''
    for item1,item2 in combinations(items_with_rating,2):
        return (item1[0],item2[0]),(item1[1],item2[1])

def calcSim(item_pair,rating_pairs):
    ''' 
        对每个item对，根据打分计算余弦距离，并返回共同打分的user个数
    '''
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    
    for rating_pair in rating_pairs:
        sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))
    return item_pair, (cos_sim,n)

def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared
    return (numerator / (float(denominator))) if denominator else 0.0

def correlation(size, dot_product, rating_sum, \
            rating2sum, rating_norm_squared, rating2_norm_squared):
    '''
        2个向量A和B的相似度
        [n * dotProduct(A, B) - sum(A) * sum(B)] /
        sqrt{ [n * norm(A)^2 - sum(A)^2] [n * norm(B)^2 - sum(B)^2] }

    '''
    numerator = size * dot_product - rating_sum * rating2sum
    denominator = sqrt(size * rating_norm_squared - rating_sum * rating_sum) * \
                    sqrt(size * rating2_norm_squared - rating2sum * rating2sum)

    return (numerator / (float(denominator))) if denominator else 0.0

def keyOnFirstItem(item_pair,item_sim_data):
    '''
        对于每个item-item对，用第一个item做key(好像有点粗暴...)
    '''
    (item1_id,item2_id) = item_pair
    return item1_id,(item2_id,item_sim_data)

def nearestNeighbors(item_id,items_and_sims,n):
    '''
        排序选出相似度最高的N个邻居
    '''
    items_and_sims.sort(key=lambda x: x[1][0],reverse=True)
    return item_id, items_and_sims[:n]

def topNRecommendations(user_id,items_with_rating,item_sims,n):
    '''
        根据最近的N个邻居进行推荐
    '''
    
    totals = defaultdict(int)
    sim_sums = defaultdict(int)

    for (item,rating) in items_with_rating:

        # 遍历item的邻居
        nearest_neighbors = item_sims.get(item,None)

        if nearest_neighbors:
            for (neighbor,(sim,count)) in nearest_neighbors:
                if neighbor != item:

                    # 更新推荐度和相近度
                    totals[neighbor] += sim * rating
                    sim_sums[neighbor] += sim

    # 归一化
    scored_items = [(total/sim_sums[item],item) for item,total in totals.items()]

    # 按照推荐度降序排列
    scored_items.sort(reverse=True)

    ranked_items = [x[1] for x in scored_items]

    return user_id,ranked_items[:n]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print >> sys.stderr, \
            "Usage: PythonItemCF <master> <file>"
        exit(-1)

    sc = SparkContext(sys.argv[1], "PythonItemCF")
    lines = sc.textFile(sys.argv[2])

    ''' 
        处理数据，获得稀疏user-item矩阵:
        user_id -> [(item_id_1, rating_1),
                   [(item_id_2, rating_2),
                    ...]
    '''
    user_item_pairs = lines.map(parseVector).groupByKey().map(
        lambda p: sampleInteractions(p[0],p[1],500)).cache()

    '''
        获取所有item-item组合对
        (item1,item2) ->    [(item1_rating,item2_rating),
                             (item1_rating,item2_rating),
                             ...]
    '''

    pairwise_items = user_item_pairs.filter(
        lambda p: len(p[1]) > 1).map(
        lambda p: findItemPairs(p[0],p[1])).groupByKey()

    '''
        计算余弦相似度，找到最近的N个邻居:
        (item1,item2) ->    (similarity,co_raters_count)
    '''

    item_sims = pairwise_items.map(
        lambda p: calcSim(p[0],p[1])).map(
        lambda p: keyOnFirstItem(p[0],p[1])).groupByKey().map(
        lambda p: nearestNeighbors(p[0],p[1],50)).collect()


    item_sim_dict = {}
    for (item,data) in item_sims: 
        item_sim_dict[item] = data

    isb = sc.broadcast(item_sim_dict)

    '''
        计算最佳的N个推荐结果
        user_id -> [item1,item2,item3,...]
    '''
    user_item_recs = user_item_pairs.map(
        lambda p: topNRecommendations(p[0],p[1],isb.value,500)).collect()