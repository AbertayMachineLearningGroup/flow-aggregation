#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:04:22 2019

@author: hananhindy
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def dataframe_drop_correlated_columns(df, threshold=0.95, verbose=False):
    if verbose:
        print('Dropping correlated columns')
        
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features
    df = df.drop(df[to_drop], axis=1)

    return df, to_drop


def dataframe_remove_const_columns(df, label_col_name = None, verbose=False):
    if verbose:
        print('Removing constant columns')
        
    width = df.shape[1]
    drop_list = []
    for col in range(width):
        num_unique = df.iloc[:,col].nunique()
        if num_unique <= 1:
            drop_list.append(col)
        
        if label_col_name != None:
            num_unique_0 = df[df[label_col_name] == 0].iloc[:, col].nunique()
            num_unique_1 = df[df[label_col_name] == 1].iloc[:, col].nunique()
       
            if (num_unique_0 <= 1 or num_unique_1 <= 1) and col != width - 1: 
                drop_list.append(col)
                
    df.drop(df.columns[drop_list], inplace = True, axis = 1)
    return df


def file_write_args(args, file_name, one_line=False):
    args = vars(args)
    
    with open(file_name, "a") as file:
        file.write('BEGIN ARGUMENTS\n')
        if one_line:
            file.write(str(args))
        else:
            for key in args.keys():
                file.write('{}, {}\n'.format(key, args[key]))
        
        file.write('END ARGUMENTS\n')

    
def plot_probability_density(array, output_file, cutoffvalue = 2):
    array[array > cutoffvalue] = cutoffvalue
    
    # Density Plot and Histogram of all arrival delays
    plt.clf()
    
    sns_plot = sns.distplot(array, hist=True, kde=True, rug=False, fit=stats.norm,
                 color = 'darkblue', 
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 4, 'label':'KDE'},
                 fit_kws={'color': 'red', 'linewidth': 4, 'label': 'PDF'})

    
    plt.xlabel("mse")
    plt.ylabel("Density")
    plt.title("PDF of mean square error (cut-off at mse = 2 s.th. mse > 2 is mapped to 2)") 
    plt.legend(loc='best')
    sns_plot.figure.savefig(output_file)
    
   
def plot_model_history(hist, output_file):
    plt.clf()
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train acc', 'val acc', 'train loss', 'val loss'], loc='upper left')
    plt.savefig(output_file)
    