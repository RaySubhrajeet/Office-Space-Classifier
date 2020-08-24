

import numpy as np
import pandas as pd

# Reading Data
officeData = pd.read_csv('ra_data_classifier.csv',encoding= 'unicode_escape')
officeData.head()

len(officeData)

officeData.shape

officeData=officeData.drop(columns=['hid'])

officeData.groupby('has_space').count()

officeData_copy=officeData.copy()
print(officeData_copy)

officeData_copy['chunk'] = officeData_copy['chunk'].str.replace('\W+', ' ').str.replace('\s+', ' ').str.strip()

officeData_copy['chunk'] = officeData_copy['chunk'].str.lower()

officeData_copy['chunk'] = officeData_copy['chunk'].str.split()

officeData_copy['chunk'].head()

officeData_copy['has_space'].value_counts() / officeData_copy.shape[0] * 100

"""# Spliting Test and Train Data"""

train_data = officeData_copy.sample(frac=0.7,random_state=1).reset_index(drop=True)
test_data = officeData_copy.drop(train_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)

train_data['has_space'].value_counts() / train_data.shape[0] * 100

train_data.shape

test_data['has_space'].value_counts() / test_data.shape[0] * 100

test_data.shape

"""# Creating list of all words"""

vocabulary = list(set(train_data['chunk'].sum()))
len(vocabulary)

print(vocabulary)

"""# Calculating frequency of words in each chunk"""

wordFrequency_per_chunk = pd.DataFrame([
    [row[0].count(word) for word in vocabulary]
    for _, row in train_data.iterrows()], columns=vocabulary)

train_data = pd.concat([train_data.reset_index(), wordFrequency_per_chunk], axis=1).iloc[:,1:]

train_data.head()

#co-efficient of cases where the word does not occur in vocabulary
alpha = 1

#total number of words in the dataset
vocabulary_size = len(vocabulary) 

#part of chunk in dataset which are labeled as renting space
renter = train_data['has_space'].value_counts()[1] / train_data.shape[0]

#part of chunk in dataset which are labeled as not renting space
non_renter = train_data['has_space'].value_counts()[0] / train_data.shape[0]

#Total number of words in chunks which are labeled as renting a space
Renter_wordCount = train_data.loc[train_data['has_space'] == 1, 'chunk'].apply(len).sum()

#Total number of words in chunks which are labeled as not renting a space
NonRent_WordCount = train_data.loc[train_data['has_space'] == 0, 'chunk'].apply(len).sum()
vocabulary_size

def probability_word_rentedChunk(word):
    if word in train_data.columns:
        return (train_data.loc[train_data['has_space'] == 1, word].sum() + alpha) / (Renter_wordCount + alpha*vocabulary_size)
    else:
        return 1

def probability_word_nonRentedChunk(word):
    if word in train_data.columns:
        return (train_data.loc[train_data['has_space'] == 0, word].sum() + alpha) / (NonRent_WordCount + alpha*vocabulary_size)
    else:
        return 1

"""# Creating classifier and predicting on test data"""

def classify(message):
    probability_renter_given_message = renter 

    probability_nonrenter_given_message = non_renter
    for word in message:
        probability_renter_given_message *= probability_word_rentedChunk(word)
        probability_nonrenter_given_message *= probability_word_nonRentedChunk(word)
    if probability_nonrenter_given_message > probability_renter_given_message:
        return 0
    elif probability_nonrenter_given_message < probability_renter_given_message:
        return 1
    else:
        return 1

classify("Office Space available for rent")

test_data['predicted'] = test_data['chunk'].apply(classify)

test_data

prediction_accuracy = (test_data['predicted'] == test_data['has_space']).sum() / test_data.shape[0] * 100

test_data.loc[test_data['predicted'] != test_data['has_space']]

print(prediction_accuracy)