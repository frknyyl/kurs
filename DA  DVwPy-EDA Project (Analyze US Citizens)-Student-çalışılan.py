#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" 
# alt="CLRSWY"></p>
# 
# ## <p style="background-color:#FDFEFE; font-family:newtimeroman; color:#9d4f8c; font-size:100%; text-align:center; border-radius:10px 10px;">WAY TO REINVENT YOURSELF</p>
# 
# ## <p style="background-color:#FDFEFE; font-family:newtimeroman; color:#060108; font-size:200%; text-align:center; border-radius:10px 10px;">Data Analysis & Visualization with Python</p>
# 
# ## <p style="background-color:#FDFEFE; font-family:newtimeroman; color:#060108; font-size:200%; text-align:center; border-radius:10px 10px;">Project Solution</p>
# 
# ![image.png](https://i.ibb.co/mT1GG7j/US-citizen.jpg)
# 
# ## <p style="background-color:#FDFEFE; font-family:newtimeroman; color:#060108; font-size:200%; text-align:center; border-radius:10px 10px;">Analysis of US Citizens by Income Levels</p>

# <a id="toc"></a>
# 
# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Content</p>
# 
# * [Introduction](#0)
# * [Dataset Info](#1)
# * [Importing Related Libraries](#2)
# * [Recognizing & Understanding Data](#3)
# * [Univariate & Multivariate Analysis](#4)    
# * [Other Specific Analysis Questions](#5)
# * [Dropping Similar & Unneccessary Features](#6)
# * [Handling with Missing Values](#7)
# * [Handling with Outliers](#8)    
# * [Final Step to make ready dataset for ML Models](#9)
# * [The End of the Project](#10)

# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Introduction</p>
# 
# <a id="0"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# One of the most important components to any data science experiment that doesn’t get as much importance as it should is **``Exploratory Data Analysis (EDA)``**. In short, EDA is **``"A first look at the data"``**. It is a critical step in analyzing the data from an experiment. It is used to understand and summarize the content of the dataset to ensure that the features which we feed to our machine learning algorithms are refined and we get valid, correctly interpreted results.
# In general, looking at a column of numbers or a whole spreadsheet and determining the important characteristics of the data can be very tedious and boring. Moreover, it is good practice to understand the problem statement and the data before you get your hands dirty, which in view, helps to gain a lot of insights. I will try to explain the concept using the Adult dataset/Census Income dataset available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult). The problem statement here is to predict whether the income exceeds 50k a year or not based on the census data.
# 
# # Aim of the Project
# 
# Applying Exploratory Data Analysis (EDA) and preparing the data to implement the Machine Learning Algorithms;
# 1. Analyzing the characteristics of individuals according to income groups
# 2. Preparing data to create a model that will predict the income levels of people according to their characteristics (So the "salary" feature is the target feature)

# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Dataset Info</p>
# 
# <a id="1"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# The Census Income dataset has 48,842 entries. Each entry contains the following information about an individual:
# 
# - **salary (target feature/label):** whether or not an individual makes more than $50,000 annually. (<= 50K, >50K)
# - **age:** the age of an individual. (Integer greater than 0)
# - **workclass:** a general term to represent the employment status of an individual. (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
# - **fnlwgt:** this is the number of people the census believes the entry represents. People with similar demographic characteristics should have similar weights.  There is one important caveat to remember about this statement. That is that since the CPS sample is actually a collection of 51 state samples, each with its own probability of selection, the statement only applies within state.(Integer greater than 0)
# - **education:** the highest level of education achieved by an individual. (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.)
# - **education-num:** the highest level of education achieved in numerical form. (Integer greater than 0)
# - **marital-status:** marital status of an individual. Married-civ-spouse corresponds to a civilian spouse while Married-AF-spouse is a spouse in the Armed Forces. Married-spouse-absent includes married people living apart because either the husband or wife was employed and living at a considerable distance from home (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
# - **occupation:** the general type of occupation of an individual. (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
# - **relationship:** represents what this individual is relative to others. For example an individual could be a Husband. Each entry only has one relationship attribute. (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
# - **race:** Descriptions of an individual’s race. (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
# - **sex:** the biological sex of the individual. (Male, female)
# - **capital-gain:** capital gains for an individual. (Integer greater than or equal to 0)
# - **capital-loss:** capital loss for an individual. (Integer greater than or equal to 0)
# - **hours-per-week:** the hours an individual has reported to work per week. (continuous)
# - **native-country:** country of origin for an individual (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">How to Installing/Enabling Intellisense or Autocomplete in Jupyter Notebook</p>
# 
# ### Installing [jupyter_contrib_nbextensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html)
# 
# **To install the current version from The Python Package Index (PyPI), which is a repository of software for the Python programming language, simply type:**
# 
# !pip install jupyter_contrib_nbextensions
# 
# **Alternatively, you can install directly from the current master branch of the repository:**
# 
# !pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master
# 
# ### Enabling [Intellisense or Autocomplete in Jupyter Notebook](https://botbark.com/2019/12/18/how-to-enable-intellisense-or-autocomplete-in-jupyter-notebook/)
# 
# 
# ### Installing hinterland for jupyter without anaconda
# 
# **``STEP 1:``** ``Open cmd prompt and run the following commands``
#  1) pip install jupyter_contrib_nbextensions<br>
#  2) pip install jupyter_nbextensions_configurator<br>
#  3) jupyter contrib nbextension install --user<br> 
#  4) jupyter nbextensions_configurator enable --user<br>
# 
# **``STEP 2:``** ``Open jupyter notebook``
#  - click on nbextensions tab<br>
#  - unckeck disable configuration for nbextensions without explicit compatibility<br>
#  - put a check on Hinterland<br>
# 
# **``Step 3:``** ``Open new python file and check autocomplete feature``
# 
# [VIDEO SOURCE](https://www.youtube.com/watch?v=DKE8hED0fow)
# 
# ![Image_Assignment](https://i.ibb.co/RbmDmD6/E8-EED4-F3-B3-F4-4571-B6-A0-1-B3224-AAB060-4-5005-c.jpg)

# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Importing Related Libraries</p>
# 
# <a id="2"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# Once you've installed NumPy & Pandas you can import them as a library:

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

plt.rcParams["figure.figsize"] = (10, 6)

sns.set_style("whitegrid")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Set it None to display all rows in the dataframe
# pd.set_option('display.max_rows', None)

# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)


# ### <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:left; border-radius:10px 10px;">Reading the data from file</p>

# In[23]:


df=pd.read_csv("/Users/furkanyayli/Downloads/adult_eda.csv")


# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Recognizing and Understanding Data</p>
# 
# <a id="3"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# ### 1.Try to understand what the data looks like
# - Check the head, shape, data-types of the features.
# - Check if there are some dublicate rows or not. If there are, then drop them. 
# - Check the statistical values of features.
# - If needed, rename the columns' names for easy use. 
# - Basically check the missing values.

# In[24]:


df.head()


# In[25]:


df.info()


# In[26]:


df.describe()


# In[27]:


df.duplicated().sum()


# In[28]:


df.drop_duplicates(keep = False, inplace = True)


# In[29]:


df.duplicated().sum()


# In[30]:


df["sex"].describe()
    


# In[31]:


df.isna().sum()


# In[78]:


df["capital-loss"].unique()


# ### 2.Look at the value counts of columns that have object datatype and detect strange values apart from the NaN Values

# In[32]:


df.describe(include="O")


# In[33]:


df.describe().T


# In[34]:


object_c=df.select_dtypes(include="object").columns
object_c


# In[35]:


cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
cat_cols


# In[36]:


df["relationship"].isna().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Univariate & Multivariate Analysis</p>
# 
# <a id="4"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# Examine all features (first target feature("salary"), then numeric ones, lastly categoric ones) separetly from different aspects according to target feature.
# 
# **to do list for numeric features:**
# 1. Check the boxplot to see extreme values 
# 2. Check the histplot/kdeplot to see distribution of feature
# 3. Check the statistical values
# 4. Check the boxplot and histplot/kdeplot by "salary" levels
# 5. Check the statistical values by "salary" levels
# 6. Write down the conclusions you draw from your analysis
# 
# **to do list for categoric features:**
# 1. Find the features which contains similar values, examine the similarities and analyze them together 
# 2. Check the count/percentage of person in each categories and visualize it with a suitable plot
# 3. If need, decrease the number of categories by combining similar categories
# 4. Check the count of person in each "salary" levels by categories and visualize it with a suitable plot
# 5. Check the percentage distribution of person in each "salary" levels by categories and visualize it with suitable plot
# 6. Check the count of person in each categories by "salary" levels and visualize it with a suitable plot
# 7. Check the percentage distribution of person in each categories by "salary" levels and visualize it with suitable plot
# 8. Write down the conclusions you draw from your analysis
# 
# **Note :** Instruction/direction for each feature is available under the corresponding feature in detail, as well.

# In[37]:


#plt.figure(figsize=(14,4))
#for i, col in enumerate(cat_cols):
#    ax = plt.subplot(1, len(cat_cols), i+1)
 #   sns.histplot(data=df, x=col, ax=ax)


# In[38]:


num_cols = [col for col in df.columns if df[col].dtypes != 'O']
num_cols


# In[39]:


sns.boxplot(x=df["education-num"])
plt.show()


# In[40]:


sns.boxplot(x=df["hours-per-week"])
plt.show()


# In[41]:


 sns.boxplot(x=df["age"])
 plt.show()


# In[42]:


fig = plt.figure(figsize=(10,10)) 
sns.boxplot(x="salary", y="age", data=df)
plt.show()


# In[43]:


fig = plt.figure(figsize=(15,10)) 
ax = sns.countplot(x="workclass", hue="salary", data=df).set_title("workclass vs count")


# In[44]:


fig = plt.figure(figsize=(25,15))
ax = sns.countplot(x="occupation", hue="salary", data=df)


# In[45]:


plt.figure(figsize=[8,5])
sns.histplot(data=df,x="age",bins=15).set(title="Distribution of age")
plt.show()


# In[46]:


fig = plt.figure(figsize=(25,15))
sns.catplot(y="education", hue="salary", kind="count",
            data=df);


# In[47]:


fig = plt.figure(figsize=(15,10)) 
ax = sns.countplot(x="relationship", hue="salary", data=df)


# In[48]:


fig = plt.figure(figsize=(15,10)) 
ax = sns.countplot(x="education-num", hue="salary", data=df)


# In[49]:


fig = plt.figure(figsize=(15,10)) 
ax = sns.countplot(x="race", hue="salary", data=df)


# In[111]:


sns.histplot(data=df,x="capital-loss",bins=20)
plt.show()


# In[ ]:





# In[ ]:





# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Other Specific Analysis Questions</p>
# 
# <a id="5"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# ### 1. What is the average age of males and females by income level?

# In[50]:


df[df['sex'] == 'Female'].groupby("salary")["age"].mean()


# In[51]:


df[df['sex'] == 'Male'].groupby("salary")["age"].mean()


# ### 2. What is the workclass percentages of Americans in high-level income group?

# In[52]:


df.head()


# In[53]:


##len(df[(df["native-country"]=="United-States")&(df["salary"]==">50K")]["native-country"])/len(df["native-country"])*100


# In[ ]:





# In[54]:


len(df["native-country"])


# In[55]:


df[(df["native-country"]=="United-States")&(df["salary"]==">50K")].groupby("workclass").count()["native-country"]


# ### 3. What is the occupation percentages of Americans who work as "Private" workclass in high-level income group?

# In[56]:


df[(df["native-country"]=="United-States")&(df["salary"]==">50K")&(df["workclass"]=="Private")].groupby("occupation").count()["native-country"]


# In[57]:


df["occupation"].value_counts()


# In[ ]:





# In[ ]:





# ### 4. What is the education level percentages of Asian-Pac-Islander race group in high-level income group?

# In[58]:


x=df[(df["race"]=='Asian-Pac-Islander')&(df["salary"]==">50K")].groupby("education").count()["race"]
x


# In[59]:


df["race"].unique()


# In[60]:


df[(df["race"]=='Asian-Pac-Islander')&(df["salary"]==">50K")].groupby("education").count()["race"].sum()


# In[61]:


##for i in x.values:
##    x[i]/276
   


# ### 5. What is the occupation percentages of Asian-Pac-Islander race group who has a Bachelors degree in high-level income group?

# In[62]:


df[(df["race"]=='Asian-Pac-Islander')&(df["salary"]==">50K")].groupby("education").count()["race"]


# In[ ]:





# 6. What is the mean of working hours per week by gender for education level, workclass and marital status? Try to plot all required in one figure.

# In[63]:


a=pd.crosstab(df['sex'], df['workclass'], 
           values=df['hours-per-week'], aggfunc=np.mean)
a


# In[64]:


b=pd.crosstab(df['education'], df['sex'], 
           values=df['hours-per-week'], aggfunc=np.mean).T


# In[65]:


c=pd.crosstab(df['marital-status'], df['sex'], 
           values=df['hours-per-week'], aggfunc=np.mean).T


# In[ ]:





# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Dropping Similar & Unneccessary Features</p>
# 
# <a id="6"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# In[84]:


df1=df.copy()
df.head()


# In[112]:


df.drop(["fnlwgt",],axis=1,inplace=True)


# In[86]:


df.head()


# In[87]:


df.corr()


# In[88]:


df.isnull().sum()


# In[91]:


df.drop(["relationship"],axis=1,inplace=True)


# In[ ]:


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


# In[101]:


high_correlated_cols(df, plot=True)


# In[ ]:





# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Handling with Missing Value</p>
# 
# <a id="7"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# In[92]:


df.isnull().sum()


# In[74]:


df1["education-num"].isna().sum()


# In[97]:


df["education-num"].fillna(df["education-num"].mean(),inplace=True)


# In[98]:


df.isnull().sum()


# In[ ]:





# In[ ]:





# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Handling with Outliers</p>
# 
# <a id="8"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# In[102]:


df.head()


# In[105]:


sns.boxplot(df['education-num'])


# In[113]:


sns.boxplot(df['capital-gain'])


# In[114]:


df.drop(["capital-loss","capital-gain"],axis=1,inplace=True)


# In[115]:


df.head()


# In[129]:


df.loc[df['education-num']<5]


# In[133]:





# ## <p style="background-color:#9d4f8c; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Final Step to Make the Dataset Ready for ML Models</p>
# 
# <a id="9"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# ### 1. Convert all features to numeric

# In[ ]:





# In[ ]:





# ### 2. Take a look at correlation between features by utilizing power of visualizing

# In[ ]:





# In[ ]:





# <a id="10"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# ## <p style="background-color:#FDFEFE; font-family:newtimeroman; color:#9d4f8c; font-size:150%; text-align:center; border-radius:10px 10px;">The End of the Project</p>
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" 
# alt="CLRSWY"></p>
# 
# ## <p style="background-color:#FDFEFE; font-family:newtimeroman; color:#9d4f8c; font-size:100%; text-align:center; border-radius:10px 10px;">WAY TO REINVENT YOURSELF</p>
# 
# ___
# 
