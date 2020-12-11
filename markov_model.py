import pandas as pd

# adding dummy start and conversion text value to visit_path based on leads_flag
def add_start_end(df,visit_path,leads_flag):
    for index, row in df.iterrows():
        i = row[visit_path]
        i.insert(0, "start")
        if row[leads_flag] == 1:
            i.append("conversion")
        else:
            i.append("exit")
        df.at[index, visit_path] = i
    return df


# transition probability
def transition_probability(df, visit_path):
    # taking all the unique sitesections to variable x
    tmp = df[[visit_path]]
    complete_sitesections = []
    for index, rows in tmp.iterrows():
        ss_list = rows[visit_path]
        for item in list(ss_list):
            complete_sitesections.append(item)
    x = set(complete_sitesections) 

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
        # creating a string by adding a comma(,) to the from_sitesection value so that we can identify if it's not the last element in path
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

    
# finding path probability
def path_probability(df, visit_path, transition_df):
    path_probability_df = df.copy()
    for index, rows in path_probability_df.iterrows():
        path = rows[visit_path]
        path_pro = 1
        for idx in range(1, len(path)):
            tmp = transition_df[(transition_df['from_sitesection'] == path[idx - 1]) & (transition_df['to_sitesection'] == path[idx])]
            path_pro = path_pro * tmp.iloc[0]['transition_probability']
        path_probability_df.at[index, 'path_probability'] = path_pro
    return path_probability_df



# finding state probability dataframe
def state_probability(df, visit_path):
    tmp = df[[visit_path]]
    complete_sitesections = []
    for index, rows in tmp.iterrows():
        ss_list = rows[visit_path]
        for item in list(ss_list):
            complete_sitesections.append(item)
    x = set(complete_sitesections) 

    state_probability = {}
    for i in x:
        state_probability.update({i:complete_sitesections.count(i)/len(complete_sitesections)})
    state_probability = pd.DataFrame(state_probability.items(),columns=['State', 'State_probability'])
    return state_probability


def convert_to_list(df, visit_path):
    from ast import literal_eval
    df[visit_path] = df[visit_path].apply(literal_eval)
    return df

def convert_to_str(df, visit_path):
    df[visit_path] = df[visit_path].astype(str)
    return df





'''
#################################################################
# custom logic for transition probability
#################################################################

# taking the path and flag columns from datax
datax18 = datax[['visit_id','visit_path','leads_flag']]

# adding dummy start and conversion text value to visit_path based on leads_flag
for index, row in datax18.iterrows():
    i = row['visit_path']
    i.insert(0, "start")
    if row['leads_flag'] == 1:
        i.append("conversion")
    else:
        i.append("exit")
    datax18.at[index, 'visit_path'] = i

# taking all the unique sitesections to variable x
tmp = datax18[['visit_path']]
complete_sitesections = []
for index, rows in tmp.iterrows():
    ss_list = rows['visit_path']
    for item in list(ss_list):
        complete_sitesections.append(item)
x = set(complete_sitesections) 



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
    # creating a string by adding a comma(,) to the from_sitesection value so that we can identify if it's not the last element in path
    denominator = "'" + row['from_sitesection'] + "', '"
    transition_df.at[index, 'string_for_denominator'] = denominator


# finding the occurance of the transition in complete dataX18 df and finding the denominator
tmp = datax18[['visit_path']].astype(str)
tmp = list(tmp['visit_path'])
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



# finding path probability
path_probability_df = datax18.copy()
for index, rows in path_probability_df.iterrows():
    path = rows['visit_path']
    path_pro = 1
    for idx in range(1, len(path)):
        tmp = transition_df[(transition_df.from_sitesection == path[idx - 1]) & (transition_df.to_sitesection == path[idx])]
        path_pro = path_pro * tmp.iloc[0]['transition_probability']
    path_probability_df.at[index, 'path_probability'] = path_pro



# adding count of browsed sitesections in visit_path
from ast import literal_eval
datax.visit_path = datax.visit_path.apply(literal_eval)
datax['sitesection_browsing_count'] = datax['visit_path'].apply(len)



# finding state probability dataframe
tmp = datax[['visit_path']]
#from ast import literal_eval
#tmp.visit_path = tmp.visit_path.apply(literal_eval)
complete_sitesections = []
for index, rows in tmp.iterrows():
    ss_list = rows['visit_path']
    for item in list(ss_list):
        complete_sitesections.append(item)

x = set(complete_sitesections) 

state_probability = {}

for i in x:
    state_probability.update({i:complete_sitesections.count(i)/len(complete_sitesections)})

state_probability = pd.DataFrame(state_probability.items(),columns=['Sitesection', 'State_probability'])

'''