#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("insurance_claim_fraud.csv")


# In[3]:


df.head(10)


# In[4]:


df.dtypes


# In[5]:


df.columns


# In[6]:


df.shape


# In[7]:


df.describe().T


# In[8]:


df.nunique()


# In[9]:


df.isnull().sum()


# In[10]:


df.drop('_c39',axis=1,inplace=True)


# In[11]:


df.info()


# # Exploratory Data Analysis: Univarient

# In[12]:


catg_features=[col for col in df.columns if df[col].dtypes=='object']
cont_features=[col for col in df.columns if df[col].dtypes!='object']


# In[13]:


print(f'Number of Categorical features: {len(catg_features)}')
print(f'Number of Continuous features: {len(cont_features)}')


# In[14]:


plt.pie([len(catg_features),len(cont_features)],labels=['Categorical','Continuous'],textprops={'fontsize':12},autopct='%1.1f%%')


# # Target Variable - Fraud Detected

# In[15]:


plt.style.use('fivethirtyeight')
ax = sns.countplot(x='fraud_reported', data=df, hue='fraud_reported')


# In[16]:


df['fraud_reported'].value_counts(dropna=False)


# In[17]:


target_df=df['fraud_reported'].value_counts()*100/df.shape[0]
target_df


# In[18]:


plt.figure(figsize=(8,6))
plt.title("Target Feature- Fraud Reported",fontdict={'fontweight':'bold','fontsize':15})
ax=sns.barplot(x=target_df.index,y=target_df.values)
plt.xlabel('Fraud Detected')
plt.ylabel('Percentage')

for p in ax.patches:
    height=p.get_height()
    width=p.get_width()
    x,_=p.get_xy()
    ax.text(x+width/2.8,height+0.5,f'{height:.2f}%')


# In[19]:


df['incident_state'].value_counts()


# In[20]:


plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(10,6))
ax = df.groupby('incident_state').fraud_reported.count().plot.bar(ylim=0)
ax.set_ylabel('Fraud reported')
plt.show()


# as the data sheet only contains data of atlantic states .

# In[22]:


df.columns


# # age vs target column

# In[23]:


sns.distplot(df['age'])


# In[24]:



plt.rcParams['figure.figsize'] = [15, 8]
table=pd.crosstab(df['age'],df['fraud_reported'])
table.div(table.sum(1),axis=0).plot(kind='bar',stacked=True)
plt.title('Age vs Fraud Reported',fontsize=15)
plt.xlabel('Age',fontsize=15)
plt.ylabel('Fraud Reported',fontsize=15)


# In[25]:


plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(18,6))
ax = df.groupby('incident_date').total_claim_amount.count().plot.bar(ylim=0)
ax.set_ylabel('Claim amount ($)')
plt.show()


#  in above plot are for the months of January and February 2015

# # policy_number

# In[26]:


df['policy_number'].nunique()


# # policy_state
# 

# In[27]:


plt.rcParams['figure.figsize'] = [10, 6]
ax= plt.style.use('fivethirtyeight')
table=pd.crosstab(df.policy_state, df.fraud_reported)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Policy State vs Fraud Reported', fontsize=12)
plt.xlabel('Policy state')
plt.ylabel('Fraud reported')
plt.show()


# In[28]:


plt.rcParams['figure.figsize']=[10,6]
plt.style.use('fivethirtyeight')   #538
table=pd.crosstab(df['policy_state'],df['fraud_reported'])
table.div(table.sum(0),axis=1).plot(kind='bar',stacked=True)
plt.title('State vs Fraud Reported',fontsize=15)
plt.xlabel('State',fontsize=15)
plt.ylabel('Fraud Reported',fontsize=15)


# # incident_state vs Tar4get variable

# In[29]:


sns.countplot(df['incident_state'],order=df['incident_state'].value_counts().index)


# In[30]:


df['incident_state'].nunique()


# In[31]:


df['incident_state'].value_counts()


# # incident_type vs fraud

# In[32]:


df['incident_type'].unique()


# In[33]:


plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(10,6))
ax = df.groupby('incident_type').fraud_reported.count().plot.bar(ylim=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
ax.set_ylabel('Fraud reported')
plt.show()


# In[34]:


df['incident_type'].value_counts()


# In[35]:


table=pd.crosstab(df['incident_type'],df['fraud_reported'])
table.div(table.sum(1),axis=0).plot(kind='bar',stacked=True)
plt.title("Incident Type  vs Fraud detected",fontsize=15)
plt.xlabel('Incident Type',fontsize=15)
plt.ylabel('Percentage ',fontsize=15)


# In[36]:


df.columns


# In[37]:


df['collision_type'].unique()


# In[38]:


df['collision_type'].value_counts(dropna=False)


# In[39]:


df['collision_type'].replace('?',np.nan,inplace=True)


# In[40]:


table=pd.crosstab(df['collision_type'],df['fraud_reported'])
table.div(table.sum(1),axis=0).plot(kind='bar',stacked=True)
plt.title("Collision Type  vs Fraud detected",fontsize=15)
plt.xlabel('Collision Type',fontsize=15)
plt.ylabel('Percentage ',fontsize=15)


# In[41]:


df['incident_severity'].unique()


# In[42]:


df['incident_severity'].value_counts(normalize=True)


# In[43]:


sns.countplot(df['incident_severity'])


# In[44]:


sns.countplot(df['incident_severity'],hue=df['fraud_reported'],palette=['#432372',"#FAAE7a"])


# # authorities_contacted vs fraud_reported

# In[46]:


df['authorities_contacted'].unique()


# In[47]:



sns.countplot(df['authorities_contacted'], order=df['authorities_contacted'].value_counts().index)


# In[48]:


df['authorities_contacted'].value_counts(normalize=True)*100


# In[49]:


sns.countplot(df['authorities_contacted'], order=df['authorities_contacted'].value_counts().index, hue=df['fraud_reported'])


# # number_of_vehicles_involved

# In[50]:


df['number_of_vehicles_involved'].unique()


# In[51]:


sns.countplot(df['number_of_vehicles_involved'], hue=df['fraud_reported'])


# # property damage
# 

# In[53]:


df['property_damage'].unique()


# In[54]:


df['property_damage'].replace('?',np.nan, inplace=True)


# In[55]:


df['property_damage'].unique()


# In[56]:


sns.countplot(df['property_damage'])


# In[57]:


sns.countplot(df['property_damage'],hue=df['fraud_reported'])


# In[58]:


df.groupby('property_damage')['fraud_reported'].value_counts()


# In[59]:


df.replace('?',np.nan,inplace=True)


# # bodily injuries
# 

# In[60]:


df['bodily_injuries'].unique()


# In[61]:


sns.countplot(df['bodily_injuries'])


# In[62]:


sns.countplot(df['bodily_injuries'],hue=df['fraud_reported'])


# In[63]:


df['witnesses'].unique()


# In[64]:


sns.countplot(df['witnesses'])


# In[65]:


sns.countplot(df['witnesses'],hue=df['fraud_reported'])


# # Vehicle Detail
# 

# # auto make
# 

# In[67]:


df['auto_make'].unique()


# In[68]:


sns.countplot(df['auto_make'],order=df['auto_make'].value_counts().index)
plt.xticks(rotation=90)


# In[69]:


sns.countplot(df['auto_make'],order=df['auto_make'].value_counts().index, hue=df['fraud_reported'])
plt.xticks(rotation=90)


# # auto model
# 

# In[70]:


df['auto_model'].unique()


# In[72]:


sns.countplot(df['auto_model'])

plt.xticks(rotation=90)


# In[73]:


sns.countplot(df['auto_model'],order=df['auto_model'].value_counts().index,hue=df['fraud_reported'])
plt.xticks(rotation=90)


# In[74]:


table=pd.crosstab(df['auto_model'],df['fraud_reported'])
table.div(table.sum(1),axis=0).plot(kind='bar',stacked=True)
plt.title('Auto Model  vs Fraud Case')


# # Insured Person 
# 

# In[75]:


df['insured_sex'].unique()


# In[76]:


plt.pie(df['insured_sex'].value_counts().values,labels=df['insured_sex'].value_counts().index,autopct='%1.2f%%')


# In[77]:



sns.countplot(df['insured_sex'],hue=df['fraud_reported'])


# # insured education level

# In[78]:


df['insured_education_level'].unique()


# In[79]:


fig = plt.figure(figsize=(10,6))
ax = sns.countplot(y = 'insured_education_level', data=df) 
ax.set_ylabel('policy_annual_premium')
plt.show()


# In[80]:


plt.rcParams['figure.figsize'] = [10, 6]
table=pd.crosstab(df.insured_education_level, df.fraud_reported)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of insured education vs Fraud reported', fontsize=12)
plt.xlabel('Insured_education_level')
plt.ylabel('Fraud reported')


# In[81]:


df['insured_occupation'].unique()


# In[82]:


table=pd.crosstab(df['insured_occupation'],df['fraud_reported'])
table.div(table.sum(1),axis=0).plot(kind='bar',stacked=True)


# # insured_relationship
# 

# In[83]:


df['insured_relationship'].unique()


# In[84]:


sns.countplot(df['insured_relationship'])


# # capital gains
# 

# In[85]:


sns.distplot(df['capital-gains'])


# In[86]:


df[cont_features]


# In[87]:


sns.distplot(df['policy_deductable'])


# In[88]:


df['policy_deductable'].unique()


# In[89]:


sns.distplot(df['policy_annual_premium'])


# In[90]:


sns.countplot(df['umbrella_limit'])
plt.xticks(rotation=45)


# In[91]:


df['umbrella_limit'].unique()


# In[92]:


df['umbrella_limit'].value_counts()


# In[93]:


sns.distplot(df['capital-gains'])


# In[94]:


sns.distplot(df['capital-loss'])


# In[95]:


df[cont_features]


# In[96]:


sns.scatterplot('age','months_as_customer',hue='fraud_reported',data=df)
plt.title('Month_as_Customer  VS  Age')
plt.xlabel('AGE',fontsize=15)


# In[97]:


sns.scatterplot('policy_deductable','policy_annual_premium',hue='fraud_reported',data=df)
plt.title('policy_deductable  VS  policy_annual_premium')
plt.xlabel('policy_deductable',fontsize=15)


# In[98]:


sns.scatterplot('capital-loss','capital-gains',hue='fraud_reported',data=df)
plt.title('capital-loss  VS  capital-gains')
plt.xlabel('capital-loss',fontsize=15)


# In[99]:


df['capital-gains'].mean() , df['capital-loss'].mean()


# In[100]:


sns.scatterplot('injury_claim','total_claim_amount',hue='fraud_reported',size='bodily_injuries',data=df)


# In[101]:


sns.scatterplot('property_claim','total_claim_amount',hue='fraud_reported',data=df)


# In[102]:


sns.scatterplot('vehicle_claim','total_claim_amount',hue='fraud_reported',size='number_of_vehicles_involved',data=df)


# In[104]:


import random

color_=['#000057','#005757','#005700','#ad7100','#008080','#575757','#003153']
cmap_=['magma','copper','crest']


# In[105]:


plt.figure(figsize=(16,50))
for i,col in enumerate(df[catg_features].columns):
    rand_col=color_[random.sample(range(6),1)[0]]
    plt.subplot(21,1,i+1)
    
    sns.countplot(data=df,x=col,color=rand_col,fill=rand_col,palette=cmap_[random.sample(range(3),1)[0]])


# In[106]:



plt.figure(figsize=(16,50))
for i,col in enumerate(df[cont_features].columns):
    rand_col=color_[random.sample(range(6),1)[0]]
    plt.subplot(6,3,i+1)
    
    sns.kdeplot(data=df,x=col,color=rand_col,fill=rand_col,palette=cmap_[random.sample(range(3),1)[0]])


# In[107]:


df.isnull().sum()


# In[108]:


df.isnull().sum()/df.shape[0]*100


# In[109]:


df[['incident_type','collision_type']]


# In[110]:


df[df['collision_type'].isnull()][['incident_type','collision_type']]


# In[111]:



df[df['collision_type'].isnull()]['incident_type'].unique()


# In[112]:


df[df['incident_type']=='Parked Car']


# In[113]:



df[df['incident_type']=='Vehicle Theft']


# In[114]:


df[df['property_damage'].isnull()]


# In[115]:


df['property_damage'].unique()


# In[116]:


df[df['police_report_available'].isnull()]


# In[117]:


Missing_coulmn=[]
for i in df.columns:
    if df[i].isnull().sum() !=0:
        df[i].fillna(df[i].mode()[0],inplace=True)


# In[118]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Target']=le.fit_transform(df['fraud_reported'])


# In[119]:



plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True,linecolor='white',linewidths=.25)


# In[120]:


df.head()


# Irreleavant columns:
#     policy_number is not required as it no help in prediction fraud case
#     policy_bind_date is not required as we have months_as_customer, how old is policy
#     insured_zip is not required as we have policy_state and many mored details for insured like sex,education,hobby,occupation,relationship
#     

# In[122]:


df['policy_csl'].unique()


# In[123]:


table=pd.crosstab(df['policy_csl'],df['fraud_reported'])
table.div(table.sum(1),axis=0).plot(kind='bar',stacked=True)


# In[124]:


df['policy_csl'].unique()


# In[125]:


df['csl_per_person']= df['policy_csl'].str.split('/',expand=True)[0]
df['csl_per_accident']= df['policy_csl'].str.split('/',expand=True)[1]


# In[126]:


df['csl_per_person'].head()


# In[127]:


df.auto_year.value_counts()


# In[128]:


df['Vehicle_Age']= 2015-df['auto_year']
df['Vehicle_Age']


# In[129]:


df['incident_hour_of_the_day'].unique()


# In[130]:


['policy_number','policy_bind_date','insured_zip','incident_date', 'incident_location','policy_csl']


# In[134]:


df[['fraud_reported','Target']]


# In[135]:


bins=[-1,5,11,16,20,24]
name=['night','Morning','afternoon','evening','midnight']
df['incident_period_of_the_day']= pd.cut(df['incident_hour_of_the_day'],bins,labels=name)


# In[136]:


df[['incident_hour_of_the_day','incident_period_of_the_day']]


# In[137]:


df.head(2)


# In[138]:


df=df.drop('incident_hour_of_the_day',axis=1)


# In[139]:


df


# In[140]:


df=df.drop('fraud_reported', axis=1)


# In[141]:


df.shape


# In[142]:


df.head(1)


# In[143]:


plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True)


# In[144]:


sns.pairplot(df)


# # Seprate Independent and Independent Variables
# 

# In[145]:


X=df.drop('Target',axis=1)
Y=df['Target']


# In[146]:


X.shape ,Y.shape


# In[147]:


catg_features=[col for col in X.columns if X[col].dtypes=='object']
cont_features=[col for col in X.columns if X[col].dtypes!='object']


# In[148]:


catg_features


# In[149]:


cont_features


# Continuous Fetaures
# 

# In[150]:


for i in cont_features:
    sns.distplot(X[i])
    plt.show()


# In[151]:


for i in cont_features:
    sns.boxplot(X[i])
    plt.show()


# In[152]:


X[cont_features].skew()


# In[153]:


missing_column=['age','policy_annual_premium','total_claim_amount','property_claim']


# In[154]:


for i in missing_column:
    IQR= X[i].quantile(.75)-X[i].quantile(.25)
    lower=X[i].quantile(.25) - (1.5 * IQR)
    upper=X[i].quantile(.75) + (1.5 * IQR)
    X[i]=np.where(X[i]<lower,lower,X[i])
    X[i]=np.where(X[i]>upper,upper,X[i])


# In[155]:


for i in cont_features:
    sns.boxplot(X[i])
    plt.show()


# Saving the Model
# 

# In[ ]:




