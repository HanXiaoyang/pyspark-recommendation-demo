#-*- coding:utf8 -*-
import pdb
import sys,os
import config

DATA_CF_LOCAL = "test_data/ratings.txt"
DATA_CF_S3 = "s3n://sparkler-data/ratings10m.txt"

def run_usercf(data):
    ''' 命令行执行基于用户的协同过滤推荐 '''

    os.system("./" + config.PYSPARK_HOME + " " + config.SPARKLER_HOME + "/userBasedRecommender.py " + config.CLUSTER_CONFIG + " " + data)

def run_itemcf(data):
    ''' 命令行执行基于物品的协同过滤推荐 '''

    os.system("./" + config.PYSPARK_HOME + " " + config.SPARKLER_HOME + "/itemBasedRecommender.py " + config.CLUSTER_CONFIG + " " + data)


if __name__ == "__main__":

    run_usercf(DATA_CF_LOCAL)

    run_itemcf(DATA_CF_LOCAL)

    run_itemcf(DATA_CF_S3)

