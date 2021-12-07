import numpy as np
import pandas as pd
import json
import csv

def analyze_data():
    data = []
    numUseful = 0
    numMarked = 0
    numMarkedNotUseful = 0
    with open('../yelp_dataset/yelp_academic_dataset_review.json') as f:
        for line in f:
            loaded_dict = json.loads(line)
            data.append(loaded_dict)
            if loaded_dict['useful'] > 0:
                numUseful += 1
            if loaded_dict['useful'] > 0 or loaded_dict['funny'] > 0 or loaded_dict['cool'] > 0:
                numMarked += 1
            if loaded_dict['useful'] == 0 and (loaded_dict['funny'] > 0 or loaded_dict['cool'] > 0):
                numMarkedNotUseful += 1
            if len(data) >= 60000:
                break
    print('Useful: ', numUseful)
    print('Marked: ', numMarked)
    print('Marked But Not Marked Useful: ', numMarkedNotUseful)


def create_csvs():
    data = []
    n = 5
    batch_size = 10000
    with open('../yelp_dataset/yelp_academic_dataset_review.json') as f:
        for line in f:
            loaded_dict = json.loads(line)
            data.append(loaded_dict)
            if len(data) >= (n+1) * batch_size:
                break

    for i in range(n+1):
        if i == n:
            with open(fr'reviews_test.csv', 'w') as f:
                writer = csv.writer(f, delimiter=',')
                for j in range(batch_size):
                    writer.writerow([data[i*batch_size + j]['useful'], data[i*batch_size + j]['text']])
                f.close()
        else:
            with open(fr'reviews_train_{i+1}.csv', 'w') as f:
                writer = csv.writer(f, delimiter=',')
                for j in range(batch_size):
                    writer.writerow([data[i*batch_size + j]['useful'], data[i*batch_size + j]['text']])
                f.close()

def processData():
    data = []
    n = 5
    batch_size = 200000
    with open('../yelp_dataset/yelp_academic_dataset_user.json') as f:
        for line in f:
            loaded_dict = json.loads(line)
            # if loaded_dict['useful'] > 0:
            #     loaded_dict['useful'] = 1
            data.append(loaded_dict)
            if len(data) >= n * batch_size:
                break

    with open('user_data_1000000.json', 'w') as f:
        json.dump(data, f)

def create_elite_data():
    with open('review_data_60000.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[['user_id', 'text']]
    df = df.sort_values(by=['user_id'])
    df = df.reset_index(drop=True)
    print(len(df), 'Reviews Loaded')

    with open('user_data_1000000.json') as f:
        users = json.load(f)
    userdf = pd.DataFrame(users)
    userdf = userdf[['user_id', 'elite']]
    userdf = userdf.sort_values(by=['user_id'])
    userdf = userdf.reset_index(drop=True)
    print(len(userdf), 'User Data Loaded')

    elite = []
    num = 0
    elitesum = 0
    p = True
    for index1, row1 in df.iterrows():
        for index2, row2 in userdf.iterrows():
            if num % 100 == 0 and p is True:
                print('Found: ', num, ' NumElites: ', elitesum, ' UsersLeft: ', len(userdf))
                p = False

            if row2['user_id'] == row1['user_id']:
                if len(row2['elite']) > 0:
                    elite.append({'elite':1, 'text':row1['text']})
                    elitesum += 1
                else:
                    elite.append({'elite':0, 'text':row1['text']})
                userdf.drop(df.index[:index2], inplace=True)
                userdf = userdf.reset_index(drop=True)
                num += 1
                p = True
                break
            elif row2['user_id'] > row1['user_id']:
                userdf.drop(df.index[:index2], inplace=True)
                userdf = userdf.reset_index(drop=True)
                break

    with open('elite_data_60000.json', 'w') as f:
        json.dump(elite, f)


def shrink_elite_data():
    with open('elite_data_60000.json') as f:
        data = json.load(f)

    newData = []
    elite = 0
    nonElite = 0
    for d in data:
        if d['elite'] == 1 and elite < 12500:
            elite += 1
            newData.append(d)
        if d['elite'] == 0 and nonElite < 12500:
            nonElite += 1
            newData.append(d)

    with open('elite_data_25000.json', 'w') as f:
        json.dump(newData, f)


# create_csvs()
# analyze_data()
# create_elite_data()
shrink_elite_data()
