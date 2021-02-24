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

def filterColumns(mask, df):
    i = 0
    output = pd.DataFrame()
    column_names = list(df.columns.values)
    for flag in mask: 
        if flag == True:
            nameStr = column_names[i]
            output[nameStr] = df.iloc[:,i]
        i = i + 1
        print(i)
    return output

def filterColumns_2(columns, df):
    i = 0
    output = pd.DataFrame()
    column_names = list(df.columns.values)
    for col in column_names: 
        if col in columns:
            nameStr = column_names[i]
            output[nameStr] = df.iloc[:,i]
        i = i + 1
        print(i)
    if len(output.columns) != len(columns):
        raise Exception("Columns not correct")
    
    return output
    
    
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
    parser.add_argument('--normal_path', default='biflow_Monday-WorkingHours_Fixed.csv')
    parser.add_argument('--attack_paths', default='biflow_Friday-WorkingHours_PortScan.csv, biflow_Tuesday-WorkingHours_SSH.csv')
    parser.add_argument('--output', default='results.csv')
    parser.add_argument('--attacker_ips', default='205.174.165.69,205.174.165.70,205.174.165.71,205.174.165.73,205.174.165.80,172.16.0.1,172.16.0.10,172.16.0.11')
    parser.add_argument('--slice_normal', type=int, default=1)
    parser.add_argument('--slice_attacks', default='1,0')
    parser.add_argument('--slice_normal_percent', type=int, default=20)
    parser.add_argument('--slice_attacks_percent', default='20, 0')
    
    parser.add_argument('--normal_slice_no', type=int, default=0)
    parser.add_argument('--slice_attacks_number', default='0,0')
    
    parser.add_argument('--choose_features', type=int, default=0)
    parser.add_argument('--selected_features', default='fwd_mean_pkt_len, bwd_mean_pkt_len, fwd_min_pkt_len, bwd_min_pkt_len, fwd_max_pkt_len, num_src_flows, src_ip_dst_prt_delta')
            
    args = parser.parse_args()
    output_file = args.output;
    
    helper.file_write_args(args, output_file)

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
    
    print("Normal= " , np.size(normal, axis = 0), file=open(output_file, "a"))
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
        print("Attack{}= ".format(count) , np.size(attack, axis = 0), file=open(output_file, "a"))
        count += 1
        XY = pd.concat([XY, attack])
        del attack
        
    column_names = list(normal.columns.values)
    del normal
    #normal = normal.values
    #stealth = stealth.values

    #XY = np.concatenate((normal, stealth), axis=0)
    width = XY.shape[1]
    length = XY.shape[0]
    X = XY.iloc[:,0:width-total_classes-1].copy()
    Y = XY.iloc[:,(width-total_classes-1)].copy()
    Y_Labels = XY.iloc[:,(width-total_classes):].copy()
    #X_df = pd.DataFrame(X)
    #sns.pairplot(X_df)
    #sns.plt.show()
    
    # Apply feature scaling to inputs only
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    Xtrans = scaler.fit_transform(X)
    
    if args.choose_features == 1:
        model = LogisticRegression(max_iter=2000) 
        rfe = RFE(estimator=model, step=1, n_features_to_select=5) # multicore
        rfe.fit(Xtrans, Y.values)

        numBestFeatures = rfe.n_features_
        featureMask = rfe.support_
        ranking = rfe.ranking_    
        #Xreduced =rfe.transform(X)
        #Xreduced_df = pd.DataFrame(Xreduced)
        Xreduced_df = filterColumns(featureMask, X)
        Xreduced_df.to_csv('rfe_Xreduced_stealth.csv', sep=',', index=False)
        Xreduced =rfe.transform(Xtrans)
        ranking_df = pd.DataFrame(ranking)
        ranking_df.to_csv('rfe_ranking_stealth.csv', sep=',', index=False)
        
        print("Columns = " , ",".join(Xreduced_df.columns), file=open(output_file, "a"))
    else:
        f_columns = args.selected_features.replace(' ', '').split(',')
        
        Xreduced = np.array(filterColumns_2(f_columns, pd.DataFrame(Xtrans, columns = X.columns)))

    #scores = rfe.grid_scores_
    #scores_df = pd.DataFrame(scores)
    #scores_df.to_csv('rfe_scores_stealth.csv', sep='\t', index=False)
    
    
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, random_state = 2017, stratify=Y)
    X = Xreduced
    Y = Y.values
    
   #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2017)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)
    
    # Try inputs: num pkts per flow, time between pkts, ave pkt size
    
    iteration = 1
    accscores = []
    recallscores = []
    for train, test in kfold.split(X, Y):
        # Initialising the ANN
        num_inputs = X.shape[1]
        #num_hidden = int((num_inputs + 1) / 2)
        num_hidden = 3
        classifier = Sequential()
        # Adding the input layer and the first hidden layer
        classifier.add(Dense(units = num_hidden, kernel_initializer = 'uniform', activation = 'relu', input_dim = num_inputs))
        # Adding the second hidden layer
        #classifier.add(Dense(units = num_hidden, kernel_initializer = 'uniform', activation = 'relu'))
        # Adding the third hidden layer
        #classifier.add(Dense(units = num_hidden, kernel_initializer = 'uniform', activation = 'relu'))
        # Adding the output layer
        classifier.add(Dense(units = total_classes, kernel_initializer = 'uniform', activation = 'sigmoid'))
        # Compiling the ANN
        classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        #classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        class_weights = class_weight.compute_class_weight('balanced', np.unique(Y[train]), Y[train])
        # Fitting the ANN to the Training set
        classifier.fit(X[train], Y_Labels.iloc[train, :], batch_size = 64, epochs = 50, class_weight=class_weights)
        
        # Making predictions and evaluating the model
        
        # Predicting the Test set results
        Y_pred = classifier.predict(X[test])
        Y_pred =  np.argmax(Y_pred, axis = 1)
        #Y_pred  = (Y_pred == Y_pred.max(axis=1)[:,None]).astype(int)
        #Y_pred = (Y_pred > 0.5)
        
        # Making the Confusion Matrix(a == a.max(axis=1)[:,None]).astype(int)
        
        cm = confusion_matrix(Y[test], Y_pred)
        cr = classification_report(Y[test],Y_pred, digits=4)
        print(cm)
        print(cr)
        print(cm, file=open(output_file, "a"))
        print(cr, file=open(output_file, "a"))
        
       
        #recall = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[1,0] + cm[2,2] + cm[2,0]) * 100
        #print("Recall = ", recall)
        #print("Recall = ", recall, file=open(output_file, "a"))
        acc = 0
        for i in range(total_classes):
                acc += cm[i, i]

        acc = (acc/np.sum(cm))*100
        print("Accuracy = ", acc)
        print("Accuracy = ", acc, file=open(output_file, "a"))
        accscores.append(acc)
        #recallscores.append(recall)
        print("iteration = ", iteration)
        print("iteration = ", iteration, file=open(output_file, "a"))
        iteration+=1
    
    print("Accuracy Ave = " , np.mean(accscores))
    print("Accuracy Std = " , np.std(accscores))
    #print("Recall Ave = " , np.mean(recallscores))
    #print("Recall Std = " , np.std(recallscores))
    
    print("Accuracy Ave = " , np.mean(accscores), file=open(output_file, "a"))
    print("Accuracy Std = " , np.std(accscores), file=open(output_file, "a"))
    #print("Recall Ave = " , np.mean(recallscores), file=open(output_file, "a"))
    #print("Recall Std = " , np.std(recallscores), file=open(output_file, "a"))

