# Artificial Neural Network

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import argparse
import operator
import helper

def load_file(file_path, is_attack = 1, attacker_ips = '', slice_data = 0, slice_percent = 20, slice_number = 0, columns_to_drop = [], label = 0, total_labels = 3):
    data = pd.read_csv(file_path)
    data = data.dropna()
  
    
    if is_attack == 0:
        data = data.loc[operator.and_(data['ip_src'].isin(attacker_ips) == False, data['ip_dst'].isin(attacker_ips) == False)]  
    else:
        data = data.loc[operator.or_(data['ip_src'].isin(attacker_ips), data['ip_dst'].isin(attacker_ips))]  
    print(np.size(data, axis = 0))
    if slice_data == 1:
        total_no = np.size(data, axis = 0)
        batch = int(total_no*(slice_percent/100))
        start = batch * slice_number
        end = batch * (slice_number + 1)
        if end > total_no:
            end = total_no - 1
        data = data.iloc[start:end, :]
        
    data.drop(columns_to_drop, axis=1, inplace=True)
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.assign(Label=label)
    for l in range(total_labels):
        strlab = 'Label{}'.format(l)
        if l == label:
            data = data.assign(x = 1)
        else:
            data = data.assign(strlab = 0)
        
        data.set_axis([*data.columns[:-1], strlab], axis=1, inplace=True)
    
    print(data.columns)
    return data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_aggregation', type=int, default=1)
    parser.add_argument('--normal_path', default='biflow_Monday-WorkingHours_Fixed_Hour_0.csv')
    parser.add_argument('--attack_paths', default='biflow_Friday-WorkingHours_PortScan.csv')
    parser.add_argument('--attacker_ips', default='205.174.165.69,205.174.165.70,205.174.165.71,205.174.165.73,205.174.165.80,172.16.0.1,172.16.0.10,172.16.0.11')
    parser.add_argument('--slice_normal', type=int, default=0)
    parser.add_argument('--slice_attacks', default='0')
    parser.add_argument('--slice_normal_percent', type=int, default=0)
    parser.add_argument('--slice_attacks_percent', default='0')
    
    parser.add_argument('--normal_slice_no', type=int, default=0)
    parser.add_argument('--slice_attacks_number', default='0')
    parser.add_argument('--number_of_features', type=int, default=5)
            
    args = parser.parse_args()

    columns_to_drop = ['ip_src', 'ip_dst', 'prt_src', 'prt_dst', 'proto']
    
    if args.drop_aggregation == 1:
        columns_to_drop.append('num_src_flows')
        columns_to_drop.append('src_ip_dst_prt_delta')
        
    attacker_ips =  args.attacker_ips.split(',')
    
    attack_paths = args.attack_paths.split(',')
    total_classes = len(attack_paths) + 1
    
    normal = load_file(
                    args.normal_path, 
                    0, 
                    attacker_ips, 
                    args.slice_normal, 
                    args.slice_normal_percent, 
                    args.normal_slice_no, 
                    columns_to_drop,
                    0, 
                    total_classes)
    
    XY = pd.concat([normal])
    
    slice_attacks = args.slice_attacks.split(',')
    slice_attacks_percent = args.slice_attacks_percent.split(',')
    slice_attacks_number = args.slice_attacks_number.split(',')
    print(slice_attacks_percent)
    count = 1
    for path in attack_paths:
        attack = load_file(
                    str.strip(path), 
                    1, 
                    attacker_ips, 
                    int(slice_attacks[count-1]), 
                    int(slice_attacks_percent[count-1]), 
                    int(slice_attacks_number[count-1]), 
                    columns_to_drop,
                    count, 
                    total_classes)
        count += 1
        XY = pd.concat([XY, attack])
        del attack
        
    column_names = list(normal.columns.values)
    del normal
    
    width = XY.shape[1]
    length = XY.shape[0]
    X = XY.iloc[:,0:width-total_classes-1].copy()
    Y = XY.iloc[:,(width-total_classes-1)].copy()
    Y_Labels = XY.iloc[:,(width-total_classes):].copy()
    
    # Apply feature scaling to inputs only
    scaler = StandardScaler()
    Xtrans = scaler.fit_transform(X)
   
    model = LogisticRegression(max_iter=2000) 
    rfe = RFE(estimator=model, step=1, n_features_to_select=1) # multicore
    rfe.fit(Xtrans, Y.values)

    numBestFeatures = rfe.n_features_
    featureMask = rfe.support_
    ranking = rfe.ranking_    
    for i in range(args.number_of_features):
        print("{} : {} ".format(i+1, X.columns[list(ranking).index(i+1)]))
        