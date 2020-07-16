#!/usr/bin/env python
# coding: utf-8

# # Starbucks Capstone Challenge
# 
# ### Introduction
# 
# This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 
# 
# Not all users receive the same offer, and that is the challenge to solve with this data set.
# 
# Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.
# 
# Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.
# 
# You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 
# 
# Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.
# 
# ### Example
# 
# To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.
# 
# However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.
# 
# ### Cleaning
# 
# This makes data cleaning especially important and tricky.
# 
# You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.
# 
# ### Final Advice
# 
# Because this is a capstone project, you are free to analyze the data any way you see fit. For example, you could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or you could build a model that predicts whether or not someone will respond to an offer. Or, you don't need to build a machine learning model at all. You could develop a set of heuristics that determine what offer you should send to each customer (i.e., 75 percent of women customers who were 35 years old responded to offer A vs 40 percent from the same demographic to offer B, so send offer A).

# # Data Sets
# 
# The data is contained in three files:
# 
# * portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
# * profile.json - demographic data for each customer
# * transcript.json - records for transactions, offers received, offers viewed, and offers completed
# 
# Here is the schema and explanation of each variable in the files:
# 
# **portfolio.json**
# * id (string) - offer id
# * offer_type (string) - type of offer ie BOGO, discount, informational
# * difficulty (int) - minimum required spend to complete an offer
# * reward (int) - reward given for completing an offer
# * duration (int) - time for offer to be open, in days
# * channels (list of strings)
# 
# **profile.json**
# * age (int) - age of the customer 
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income
# 
# **transcript.json**
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours since start of test. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record
# 
# **Note:** If you are using the workspace, you will need to go to the terminal and run the command `conda update pandas` before reading in the files. This is because the version of pandas in the workspace cannot read in the transcript.json file correctly, but the newest version of pandas can. You can access the termnal from the orange icon in the top left of this notebook.  
# 
# You can see how to access the terminal and how the install works using the two images below.  First you need to access the terminal:
# 
# <img src="pic1.png"/>
# 
# Then you will want to run the above command:
# 
# <img src="pic2.png"/>
# 
# Finally, when you enter back into the notebook (use the jupyter icon again), you should be able to run the below cell without any errors.

# In[2]:


import pandas as pd
import numpy as np
import math
import json
get_ipython().run_line_magic('matplotlib', 'inline')

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


# In[748]:


# Added after first cell

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import cross_validate


# In[424]:


portfolio


# In[44]:


transcript.tail(10)


# In[6]:


portfolio.shape, profile.shape, transcript.shape


# In[ ]:





# ### Cleaning for portfolio dataset
# 

# In[20]:


portfolio.head()


# In[ ]:


# Now we will create new columns for portfolio dataset which shows us channels seperatly
# We will use it after cleaning all datasets..

channels = ['web','email','mobile','social']

for channel in channels:
    portfolio[channel] = portfolio.channels.apply(lambda x: 1 if channel in x else 0)
    


# In[48]:


portfolio


# In[40]:


portfolio.drop('channels',axis=1,inplace=True)
portfolio.head()


# In[34]:


pd.get_dummies(portfolio['offer_type'])


# In[45]:


# Duration period is day for portfolio, while hours for transcript
# So I prefer to continue with hours value which can be more detail for our model efficiency.
# Than rename portfolio id column with offer_id, to avoid possible confision.

portfolio['duration']*=24


# In[50]:


portfolio.rename(columns={'id': 'offer_id'}, inplace=True)


# In[52]:


portfolio


# In[ ]:





# ### Cleaning for profile dataset

# In[ ]:


# I will encode the user id, It isn't necessery anyway..


# In[144]:



def user_mapper():
    coded_dict = dict()
    cter = 1
    user_encoded = []
    
    for val in profile['id']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        user_encoded.append(coded_dict[val])
    return user_encoded

user_encoded = user_mapper()

profile['user_id'] = user_encoded

# show header
profile.head()


# In[145]:


# So we can see that both profile and transaction id's has same amount of users.. 17000
# We can encode the id to user id for both datasets for better understanding
profile.id.nunique(),transcript.person.nunique()


# In[146]:


profile.isnull().sum()

# We need to check out if the values for the same users or not..


# In[147]:


profile[profile['gender'].isnull()]

# It seems they all the same user id, so we can drop them since we dont have any income or gender data..


# In[149]:


profile[profile.age==118]


# In[150]:


profile_new=profile.dropna()
profile_new.shape


# In[151]:


profile_new.head()


# In[154]:



profile_new['became_member_on'] = pd.to_datetime(profile.became_member_on, format='%Y%m%d')


# In[155]:


profile_new.head()


# In[ ]:


# Now we can calculate for how long user is member in starbucks..
# we can assume maximum date as 2019-01-01 and set the membership since as 'member_since_days'


# In[156]:


profile_new.became_member_on.max()


# In[173]:


max_day=pd.to_datetime('20190101', format='%Y%m%d')


# In[179]:


profile_new['member_since_days']=(max_day-profile_new.became_member_on).dt.days


# In[259]:


profile_new.drop('became_member_on',axis=1, inplace=True)


# In[ ]:


# Change 'id' column with 'customer_id' for merging easily 


# In[181]:


profile_new.rename(columns={"id": "customer_id"},inplace=True)


# In[260]:


profile_new.head()


# In[188]:


# Customer Age hist
plt.figure()
user_age = profile_new['age'].plot(kind='hist', bins=30, title='Customer Age Distribution')
user_age.set_xlabel("Customer Age")

# Display Histogram of the days being member
plt.figure()
memberdays = profile_new['member_since_days'].plot(kind='hist', bins=50, title='Member Since')
memberdays.set_xlabel("Days")
    
# Display Histogram of User Income
plt.figure()
user_income = profile_new['income'].plot(kind='hist', bins=20, title='Customer Income')
user_income.set_xlabel("Income")


# In[ ]:





# In[ ]:





# ### Cleaning for Transcript dataset

# In[ ]:





# In[ ]:


# At first we will create seperated offer_id and amount columns..
# Than drop the value column since we no longer need it
# Rename the person column as customer_id for merging easily
# We have some users who we dont have any information about( age gender etc. ) We will drop that rows
# At last, we will rename transcript dataset as transcript_new for easily future calculations


# In[199]:


transcript.head()


# In[220]:


transcript['offer_id'] = transcript.value.apply(lambda x: x[list(x.keys())[0]] if list(x.keys())[0] in ['offer_id','offer id'] else None)
transcript['amount'] = transcript.value.apply(lambda x: x['amount'] if 'amount' in x.keys() else None)


# In[221]:


transcript=transcript.drop(columns='value')


# In[233]:


transcript.rename(columns={'person': 'customer_id'}, inplace=True)


# In[237]:


transcript.head()


# In[239]:


transcript.isnull().sum()


# In[245]:


# We have 33772 person who we dont have any information about( age gender etc. )

len(transcript)-len(transcript[transcript.customer_id.isin(profile_new.customer_id)])


# In[246]:


transcript_new=transcript[transcript.customer_id.isin(profile_new.customer_id)]


# In[261]:


transcript_new.duplicated().sum()


# In[262]:


transcript_new.drop_duplicates(inplace=True)


# In[ ]:


# It will be more understandable if we change the offer_ids more readable way. So let's encode it.
# And we can rename it like other datasets, portfolio_new..


# In[654]:


portfolio.head()


# In[656]:


portfolio_new=portfolio.copy()


# In[665]:


def offer_mapper():
    coded_dict = dict()
    cter = 1
    offer_encoded = []
    
    for val in portfolio_new['offer_id']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        offer_encoded.append(coded_dict[val])
    return offer_encoded

offer_encoded = offer_mapper()

for i in range(len(offer_encoded)):
    portfolio_new['offer_id'][i]=str(offer_encoded[i])+'_offer'
    
# show header
portfolio_new.head()


# In[667]:


portfolio_new.rename(columns={'offer_id': 'new_offer_id'}, inplace=True)


# In[668]:


portfolio_new['offer_id']=portfolio['offer_id']


# In[689]:


portfolio_new.head()


# In[690]:


profile_new.head()


# In[691]:


transcript_new.head()


# In[ ]:


# Now we have transaction_new, profile_new and portfolio_new datasets..


# 
# # LET'S GET TO WORK

# In[755]:


# At firts will merge profile&transcript and portfolio&transcript datasets for future usage..


# In[692]:


profile_transcript=profile_new.merge(transcript_new, on='customer_id')


# In[693]:


profile_transcript.head()


# In[ ]:





# In[694]:


portfolio_transcript=pd.merge(transcript_new,portfolio_new, on='offer_id',how='left')


# In[695]:


portfolio_transcript.head()


# In[696]:


portfolio_transcript.event.value_counts()


# In[697]:


# We can see that tx didn't happens or user can't see if offer just receives..
# Now we can drop that particular rows(offer received)
# than create different datasets with all conditions except offer received condition.


# In[698]:


portfolio_transcript=portfolio_transcript[portfolio_transcript['event']!='offer received']


# In[699]:


offer_viewed=portfolio_transcript[portfolio_transcript['event']=='offer viewed']
offer_completed=portfolio_transcript[portfolio_transcript['event']=='offer completed']
transaction=portfolio_transcript[portfolio_transcript['event']=='transaction']


# In[700]:


offer_viewed.head()


# In[701]:


offer_completed.head()


# In[702]:


transaction.head()


# In[ ]:





# In[ ]:


# Than we will calculate the time of customer became a member when the offer sent to person and between time after receiving and tx happens
# create new columns for this calculations. Name it 'customer_offers' dataframe
# Than merge customer_offers dataset with portfolio new for seeing all datas together..
# After merging, we have no longer need to keep 'offer_id' column which is long and complex data.So we will drop it.


# In[686]:


offers = []

offers_viewed = portfolio_transcript[portfolio_transcript.event == 'offer viewed']
offers_completed = portfolio_transcript[portfolio_transcript.event == 'offer completed']
transactions = portfolio_transcript[portfolio_transcript.event == 'transaction']

for index, offer in offers_viewed.iterrows():
    customer_id = offer.customer_id
    start_time = offer.time
    end_time = int(start_time + offer.duration)

    within_time = (transactions.time <= end_time) & (transactions.time >= start_time)
    transaction_amount = transactions[(within_time) & (transactions.customer_id == customer_id)]['amount'].sum()

    within_time = (offers_completed.time <= end_time) & (offers_completed.time >= start_time)
    n_completed = len(offers_completed[(offers_completed.customer_id == customer_id) & (offers_completed.offer_id == offer.offer_id) & within_time])
    
    customer_offer = {
        'customer_id': customer_id,
        'offer_id': offer.offer_id,
        'start_time': start_time,
        'difficulty': offer.difficulty,
        'duration': offer.duration,
        'type': offer.offer_type,
        'reward': offer.reward,
        'web': offer.web,
        'email': offer.email,
        'mobile': offer.mobile,
        'social': offer.social,
        'transaction_amount': transaction_amount,
        'completed': n_completed
        }
    
    offers.append(customer_offer)

customer_offers = pd.DataFrame(offers)


# In[704]:


customer_offers.head()


# In[705]:


portfolio_new.head()


# In[706]:


# We can change offer_id's to numeric values for simpler looking, we will merge portfolio_new and customer_offers datasets


# In[707]:


customer_offers=customer_offers.merge(portfolio_new[['new_offer_id', 'offer_id']], on='offer_id')


# In[709]:


customer_offers.drop('offer_id',axis=1,inplace=True)


# In[710]:


customer_offers.head()


# In[711]:


profile_new.head()


# In[712]:


customer_profile=customer_offers.merge(profile_new,on='customer_id')


# In[714]:


# We have no longer need to keep customer_id since we have new user id column for this dataset..
customer_profile.drop(['customer_id','user_id'],axis=1,inplace=True)


# In[737]:


customer_profile.rename(columns={'new_offer_id': 'offer_id'}, inplace=True)


# In[738]:


# OUR DATASET IS READY FOR BUILDING MODEL ON IT. LET'S START BUILDING A MODEL


# In[739]:


customer_profile.head()


# ### MODEL BUILDING

# In[ ]:





# In[ ]:


# We need to split the data to features(X) and target(y) datasets
# We will get dummy variables for X because we have gender offers and type of offer datasets which are need to get dummy variables.
# Set the types as float for all values in columns before fitting to 'standardscaler()' 


# In[ ]:





# In[740]:


X=customer_profile.drop('completed',axis=1)  # Features
y=customer_profile['completed']              # Target


# In[741]:


X=pd.get_dummies(X)


# In[742]:


X.columns


# In[743]:


X['start_time'] = X['start_time'].astype(float)
X['age'] = X['age'].astype(float)
X['member_since_days'] = X['member_since_days'].astype(float)
X['offer_id_1_offer'] = X['offer_id_1_offer'].astype(float)
X['offer_id_2_offer'] = X['offer_id_2_offer'].astype(float)
X['offer_id_3_offer'] = X['offer_id_3_offer'].astype(float)
X['offer_id_4_offer'] = X['offer_id_4_offer'].astype(float)
X['offer_id_5_offer'] = X['offer_id_5_offer'].astype(float)
X['offer_id_6_offer'] = X['offer_id_6_offer'].astype(float)
X['offer_id_7_offer'] = X['offer_id_7_offer'].astype(float)
X['offer_id_8_offer'] = X['offer_id_8_offer'].astype(float)
X['offer_id_9_offer'] = X['offer_id_9_offer'].astype(float)
X['offer_id_10_offer'] = X['offer_id_10_offer'].astype(float)
X['type_bogo'] = X['type_bogo'].astype(float)
X['type_discount'] = X['type_discount'].astype(float)
X['type_informational'] = X['type_informational'].astype(float)
X['gender_F'] = X['gender_F'].astype(float)
X['gender_M'] = X['gender_M'].astype(float)
X['gender_O'] = X['gender_O'].astype(float)


# In[ ]:


# Now it is time to split the data to train and test split.
# Then build a pipeline for seeing best classification alghorithm..
# We will choose best test score and apply to our data and then analyse it.


# In[ ]:





# In[749]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

print("Training set sample size is {} .".format(X_train.shape[0]))
print("Testing set sample size is {} .".format(X_test.shape[0]))


# In[750]:


clfs = []

pipeline = Pipeline([
    ('normalizer', StandardScaler()), 
    ('clf', KNeighborsClassifier())
]) 

clfs = []
clfs.append(KNeighborsClassifier())
clfs.append(DecisionTreeClassifier(random_state=0))
clfs.append(RandomForestClassifier(n_estimators=100, random_state=0))
clfs.append(GradientBoostingClassifier(random_state=0))
clfs.append(SVC(random_state=0, gamma='scale'))

for classifier in clfs:
    pipeline.set_params(clf = classifier)
    scores = cross_validate(pipeline, X_train, y_train, cv=5)
    print('---------------------------------')
    print(str(classifier))
    print('-----------------------------------')
    for key, values in scores.items():
            print(key,' => ', values.mean())


# In[ ]:


# Best estimators are RandomForestClassifier and GradientBoostingClassifier with %88 estimation percentage
# I prefer to choose RandomForestClassifier for my dataset, so let's build a pipeline with 100 estimators.


# In[752]:


pipeline = Pipeline([
    ('normalizer', StandardScaler()), 
    ('rfc', RandomForestClassifier(n_estimators=100, random_state=0))
])             

pipeline.fit(X_train, y_train)
print("Test dataset accuracy score = %3.2f" %(pipeline.score(X_test,y_test)))


# In[ ]:





# ## DATA ANALYSIS

# ## Q1: Which features are most important to prefer which user be chosen?

# In[756]:


def feature_plot(strength, X_train, y_train):
    
    # Display the five most important features
    indexes = np.argsort(strength)[::-1]
    columns = X_train.columns.values[indexes[:5]]
    values = strength[indexes][:5]

    # Creat the plot
    fig = plt.figure(figsize = (10,6))
    plt.title("Normalized Strength of first 5 Predictive Features", fontsize = 16)
    plt.bar(np.arange(5), values, width = 0.6, align="center", color = '#EE7621',           label = "Feature Strength")
    plt.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#0000EE',           label = "Cumulative Strength")
    plt.xticks(np.arange(5), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Strength", fontsize = 12)
    plt.xlabel("Feature Name", fontsize = 12)
    
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.show()  
    
    
model = RandomForestClassifier(n_estimators=100,random_state=0).fit(X_train, y_train)
strength = model.feature_strength_
feature_plot(strength, X_train, y_train)


# It seems like transaction_amount is the most important one having the most predictive power. Membership duration, starting time(suprisely) and income from customer then age follows it.
# start_time is the interesting one, maybe this is because offers are valid within a specific period of time and timely transactions matter to achieve completing the offer or not.
# Income is an important factor which shows economic power of an individual.
# 
# 

# In[ ]:





# ## Q2: Which gender is completing offers the most? and

# In[ ]:





# In[772]:


male_total=len(customer_profile[customer_profile.gender=='M'])
female_total=len(customer_profile[customer_profile.gender=='F'])
other_total=len(customer_profile[customer_profile.gender=='O'])


# In[773]:


male_completed=len(customer_profile[(customer_profile.completed == 1) & (customer_profile.gender=='M')])
female_completed=len(customer_profile[(customer_profile.completed == 1) & (customer_profile.gender=='F')])
other_completed=len(customer_profile[(customer_profile.completed == 1) & (customer_profile.gender=='O')])


# In[788]:


male_rate=male_completed/male_total
female_rate=female_completed/female_total
other_rate=other_completed/other_total
print('Convertion rate for male customers:',male_rate,'\nConvertion rate for female customers:',female_rate,'\nConvertion rate for other gender customers:',other_rate)


# In[754]:


customer_profile[customer_profile.completed == 1].gender.value_counts().plot(kind='bar')


# Men are completing offers more than women for total amount. Maybe this is because they drink more or they are more careful on following offers.
# But for convertion rate of customers, we can say that female and other gender customers are better and profitable target for Starbucks Company..

# In[ ]:




