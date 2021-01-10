# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:51:18 2020

@author: Ishan Jain
"""

import pandas as pd
from ast import literal_eval


def add_start_end(df,visit_path,leads_flag):
    """
    Adds dummy start and conversion text value to visit_path based on leads_flag

    :df: dataframe, dataframe that needs to be updated
    :visit_path: str, name of column which contains path data
    :leads_flag: str, name of column that has conversion flagged as 1

    Returns
    :df: dataframe, updated dataframe with start, conversion and exit values
    """

    for index, row in df.iterrows():
        i = row[visit_path]
        i.insert(0, "start")
        if row[leads_flag] == 1:
            i.append("conversion")
        else:
            i.append("exit")
        df.at[index, visit_path] = i
    return df


def get_unique_points(df, visit_path):
    """
    Returns the set of sitesections.

    :df: dataframe, dataframe that has path
    :visit_path: str, name of column which contains path data
    
    Returns
    :x: set, unique values of sitesections in complete dataframe
    """
    tmp = df[[visit_path]]
    complete_sitesections = []
    for index, rows in tmp.iterrows():
        ss_list = rows[visit_path]
        for item in list(ss_list):
            complete_sitesections.append(item)
    x = set(complete_sitesections)
    return x, complete_sitesections



def transition_probability(df, visit_path):
    """
    To find the transition probability.

    :df: dataframe, dataframe that has path
    :visit_path: str, name of column which contains path data
    
    Returns
    :transition_df: dataframe, dataframe that contrains the transition probability
    """

    # taking all the unique sitesections to variable x
    x, complete_sitesections = get_unique_points(df, visit_path) 

    # creating dummy df for transition probability
    transition_df = []
    for item1 in x:
        for item2 in x:
            if item1 != item2:
                transition_df.append([item1,item2])
    transition_df = pd.DataFrame(transition_df,columns=['from_sitesection', 'to_sitesection'])
    transition_df = transition_df[transition_df['from_sitesection'] != "conversion"]
    transition_df = transition_df[transition_df['to_sitesection'] != "start"]
    transition_df = transition_df[transition_df['from_sitesection'] != "exit"]


    # cocatenatng the transition string
    for index, row in transition_df.iterrows():
        path_in_string = "'" + row['from_sitesection'] + "', '" + row['to_sitesection'] + "'"
        transition_df.at[index, 'path_string'] = path_in_string
        # creating a string by adding a comma(,) to the from_sitesection value
        # so that we can identify if it's not the last element in path
        denominator = "'" + row['from_sitesection'] + "', '"
        transition_df.at[index, 'string_for_denominator'] = denominator

    # finding the occurance of the transition in complete df and finding the denominator
    tmp = df[[visit_path]].astype(str)
    tmp = list(tmp[visit_path])
    for index, row in transition_df.iterrows():
        co = 0
        denominator_counter = 0
        for item in tmp:
            co = co + item.count(row['path_string'])
            # finding the denominator for probability as mentioned in the documentation on -
            # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.679.2662&rep=rep1&type=pdf
            if row['string_for_denominator'] in item:
                denominator_counter = denominator_counter + 1
        transition_df.at[index, 'transition_occurance'] = co
        transition_df.at[index, 'denominator'] = denominator_counter
    transition_df['transition_probability'] = transition_df['transition_occurance']/transition_df['denominator']
    del transition_df['transition_occurance']
    del transition_df['denominator']
    del transition_df['path_string']
    del transition_df['string_for_denominator']
    return transition_df



def path_probability(df, visit_path, transition_df):
    """
    To find the path probability.

    :df: dataframe, dataframe that has path column
    :visit_path: str, name of column which contains path data
    :transition_df: dataframe, dataframe that contrains the transition probability

    Returns
    :path_probability_df: dataframe, dataframe that contains probability of paths
    """
    path_probability_df = df.copy()
    for index, rows in path_probability_df.iterrows():
        path = rows[visit_path]
        path_pro = 1
        for idx in range(1, len(path)):
            tmp = transition_df[(transition_df['from_sitesection'] == path[idx - 1]) & (transition_df['to_sitesection'] == path[idx])]
            path_pro = path_pro * tmp.iloc[0]['transition_probability']
        path_probability_df.at[index, 'path_probability'] = path_pro
    return path_probability_df



def state_probability(df, visit_path):
    """
    To find the state probability.

    :df: dataframe, dataframe that has path column
    :visit_path: str, name of column which contains path data

    Returns
    :state_probability: dataframe, dataframe that contains probability of being on a state
    """

    # taking all the unique sitesections to variable x
    x, complete_sitesections = get_unique_points(df, visit_path) 
    state_probability = {}
    for i in x:
        state_probability.update({i:complete_sitesections.count(i)/len(complete_sitesections)})
    state_probability = pd.DataFrame(state_probability.items(),columns=['State', 'State_probability'])
    return state_probability



def convert_to_list(df, visit_path):
    """
    To convert the path column to list type.

    :df: dataframe, dataframe that has path column
    :visit_path: str, name of column which contains path data

    Returns
    :df: dataframe, dataframe that contains path column as list type
    """
    df[visit_path] = df[visit_path].apply(literal_eval)
    return df



def convert_to_str(df, visit_path):
    """
    To convert the path column to string type.

    :df: dataframe, dataframe that has path column
    :visit_path: str, name of column which contains path data

    Returns
    :df: dataframe, dataframe that contains path column as string type
    """
    df[visit_path] = df[visit_path].astype(str)
    return df
