#!/usr/bin/env python
# coding: utf-8

# In[266]:


# pandas - basics 
#Each col in a dataframe is a series
#pandas provides a lot of functionalities, each of them a method you can apply to a DataFrame or Series
#As methods are functions do not forget to use the ()
import pandas as pd

df=pd.DataFrame({"name":["anu","mr.oven" , "allen" , "mr.william ", "miss.linda"],
             
             'Age':[22,45,78,10,89],
             'Sex':['f','m','m','m','f']
             })


# In[267]:


df


# In[268]:


df["Age"]


# In[269]:


type(['Age'])


# In[270]:


ages=pd.Series(['a','b','c','d'])


# In[271]:


type(ages)


# In[272]:


ages


# In[276]:


df['Age'].min()


# In[38]:


df.describe()


# In[39]:


#Import the package, aka import pandas as pd

#A table of data is stored as a pandas DataFrame

#Each column in a DataFrame is a Series

#You can do things by applying a method to a DataFrame or Series

# attributes of a dataframe do not need brackets where as a method needs bracket.


# In[40]:


#to read a csv file


# In[4]:


import pandas as pd


# In[5]:


titanic=pd.read_csv("train.csv")


# In[9]:


titanic.dtypes


# In[17]:


titanic.to_excel("titan.xlsx",sheet_name = "pass",index=False)


# In[18]:


titanic.info()


# In[75]:


#Getting data in to pandas from many different file formats or data sources is supported by read_* functions.

#Exporting data out of pandas is provided by different to_*methods.

#The head/tail/info methods and the dtypes attribute are convenient for a first check.

# to select a single col use square brackets[] with the col name 

#To select multiple columns, use a list of column names within the selection brackets []

#isin() is a function 

#When combining multiple conditional statements, each condition must be surrounded by parentheses (). 
#Moreover, you can not use or/and but need to use the or operator | and the and operator &.

#notna() is a conditional function returns true for each row with values that are not null.

# for selecting only specific rows and colums we should use loc/iloc operators in front of the square bracket

#When using loc/iloc, the part before the comma

#You can assign new values to a selection based on loc/iloc.

#nunique - will give a count of unique values in a particular column


# In[21]:


titanic


# In[380]:


age_sex=titanic[['Sex','Age']]


# In[50]:


age_sex.shape


# In[ ]:


#filtering specific rows from a dataframe


# In[57]:


above_35=titanic [titanic['Age']>35]


# In[59]:


above_35.shape


# In[66]:


class2_3=titanic[titanic['Pclass'].isin([2,3])]


# In[64]:


class2_3


# In[68]:


titanic[titanic['Cabin'].notna()]


# In[76]:


titanic.loc[titanic['Age']>35,"Name"]


# In[79]:


titanic.iloc[9:25,2:5]


# In[82]:


titanic.iloc[0:3,3]='anu'


# In[83]:


titanic


# In[111]:


titanic['Cabin']


# In[128]:


titanic['Ticket'].nunique()


# In[129]:


# creating plots in pandas
#plot() method works for both series and dataframes


# In[151]:


import pandas as pd
import matplotlib.pyplot as plt


# In[248]:


airq=pd.read_csv("monthly-averages_csv.csv")


# In[249]:


airq


# In[250]:


airq["London_Mean_Roadside_Ozone"].plot()


# In[251]:


#alpha indicates the transperency of the dots
airq.plot(x="London_Mean",y="London_Mean_Roadside_Ozone",alpha=0.5)


# In[660]:


airq.plot.box()


# In[253]:


airq.plot.area(figsize=(12,4),subplots=True)


# In[277]:


#to create new col from the existing col
#To create a new column, use the [] brackets with the new column name at the left side of the assignment.


# In[278]:


import pandas as pd


# In[280]:


como=pd.read_csv("commodity_price.csv")


# In[281]:


como


# In[325]:


como["avg_price"]=(como["min_price"]+como["max_price"])*5


# In[326]:


como


# In[329]:


como['new_model_price']=como['modal_price']*2


# In[330]:


como


# In[341]:


como["comparison"]=(como["max_price"]/como["new_model_price"])


# In[342]:


como


# In[349]:


# apply functions in pandas and numpy
import pandas as pd
import numpy as np


# In[350]:


df=pd.DataFrame([[4,9]]*3,columns=['a','b'])


# In[351]:


df


# In[352]:


df.apply(np.sqrt)


# In[353]:


df.apply(np.sum,axis=0)


# In[354]:


df.apply(np.sum,axis=1)


# In[359]:


df.apply(lambda x:[1,2],axis=1)# return  a list like will result in series


# In[361]:


df.apply(lambda x:[1,2],axis=1,result_type='expand') # will expand list like results to columns of a dataframe


# In[364]:


#summary satistics
#The .plot.* methods are applicable on both Series and DataFrames.
#By default, each of the columns is plotted as a different element (line, boxplot,…).
#Any plot created by pandas is a Matplotlib object.
#Instead of predefined statistics ,specific combinations of aggregating statistics for a given col can be 
#done using DataFrame.agg() method
#The rename() function can be used for both row labels and column labels. Provide a dictionary with the keys 
#the current names and the values the new names to update the corresponding names.
#rename(columns=str.upper) - used to covert col to upper or lower case
#Instead of the predefined statistics, specific combinations of aggregating statistics 
#,for given columns can be defined using the DataFrame.agg() method:
import pandas as pd


# In[370]:


newtita=pd.read_csv("train.csv")


# In[371]:


newtita


# In[386]:


newtita["Age"].mean()


# In[376]:


newtita[["Age","Fare"]].median()


# In[377]:


newtita[["Age","Fare"]].describe()


# In[379]:


newtita.agg({"Age":["min","max","median","skew"],
           "Fare":["min","max","median","skew"]}) 


# In[388]:


#RENAME THE COL
newtita.rename(columns={ "PassengerId":"ID",
                        "Survived": "sur",
                        "Pclass": "Passclass"
                       })


# In[390]:


newtita


# In[392]:


newtita.rename(columns=str.upper)


# In[393]:


#how to calculate summary statistics
#These operations generally exclude missind data
#split-apply-combine pattern:
#column containing numerical columns by passing numeric_only=True
#the value_counts() method counts the number of records for each category in a column.


# In[400]:


#aggregate statistics
newtita["Age"].mean()


# In[401]:


newtita[["Age","Fare"]].mean()


# In[402]:


newtita[["Age","Fare"]].median()


# In[403]:


newtita[["Age","Fare"]].describe()


# In[408]:


#aggregate based on grouped by category
newtita[["Sex",'Age']].groupby("Sex").median()


# In[409]:


newtita.groupby("Sex").mean(numeric_only=True)


# In[411]:


newtita.groupby("Sex")["Age"].median()


# In[413]:


newtita.groupby(["Pclass","Sex"])["Fare"].mean()


# In[418]:


newtita["Pclass"].value_counts()


# In[424]:


newtita["Pclass"].count()


# In[430]:


newtita["Cabin"].dropna()


# In[431]:


#reshape the layout of tables
#With DataFrame.sort_values(), the rows in the table are sorted according to the defined column(s). 
#The index will follow the row order.
#The pivot() function is purely reshaping of the data: a single value for each index/column combination is required.
#The pandas.melt() method on a DataFrame converts the data table from wide format to long format. 
#The column headers become the variable names in a newly created column.
#Sorting by one or more columns is supported by sort_values.
#The pivot function is purely restructuring of the data, pivot_table supports aggregations.
#The reverse of pivot (long to wide format) is melt (wide to long format).
#Multiple tables can be concatenated both column-wise and row-wise using the concat function.
#For database-like merging/joining of tables, use the merge function.


# In[437]:


newtita.sort_values(by=["Age"])


# In[441]:


newtita.sort_values(by=["Age","Pclass"],ascending = False)


# In[442]:


import pandas as pd


# In[447]:


pd.read_csv("commodity_price.csv")


# In[503]:


como


# In[471]:


#creating a new subset of the present dataset 
newcomo=como[como["state"]=='Karnataka']


# In[472]:


newcomo


# In[514]:


como.pivot(columns="variety",values="s_no")


# In[522]:


#combining multiple table
n=pd.concat([newcomo,como],axis=1)


# In[523]:


n


# In[524]:


#how to handle time series data sets
#By applying the to_datetime function, pandas interprets the strings and convert these to datetime 
#(i.e. datetime64[ns, UTC]) objects
#Valid date strings can be converted to datetime objects using to_datetime function or as part of read functions.

#Datetime objects in pandas support calculations, logical operations and convenient date-related properties using #the dt accessor.

#A DatetimeIndex contains these date-related properties and supports convenient slicing.

#Resample (time based grouping) is a powerful method to change the frequency of a time series.


# In[590]:


m=pd.read_csv("data sets/Gold.csv")


# In[591]:


m


# In[548]:


m.columns


# In[585]:


m.shape


# In[586]:


for column in m:
    print(column)


# In[594]:


m["DATE"] = pd.to_datetime(m["DATE"])


# In[595]:


m["DATE"] 


# In[608]:


O=pd.DataFrame({
                "city":"Paris",
               "country":"FR",
                "datetime":["2019-06-21 00:00:00+00:00","2019-06-20 23:00:00+00:00","2019-06-20 22:00:00+00:00"],
                "value":"FR04014",
                "unit":"no2 20.0 µg/m³"
               })


# In[609]:


O


# In[610]:


O["datetime"]=pd.to_datetime(O["datetime"])


# In[611]:


O["datetime"]


# In[615]:


p=pd.read_csv("data sets/commodity_price.csv",parse_dates=["datetime"])


# In[617]:


#how to manipulate textual data
#Using the Series.str.split() method, each of the values is returned as a list of 2 elements. 
#The first element is the part before the comma and the second element is the part after the comma.
#The string method Series.str.contains() checks for each of the values in the column Name if the string contains the word Countess 
#and returns for each of the values True
#String methods are available using the str accessor.

#String methods work element-wise and can be used for conditional indexing.

#The replace method is a convenient method to convert values according to a given dictionary.
newtita


# In[623]:


newtita["Cabin"].str.lower()


# In[624]:


newtita["Name"].str.split(",")


# In[632]:


newtita["surname"]=newtita["Name"].str.split(",").str.get(1)


# In[633]:


newtita["surname"]


# In[634]:


newtita["surname"]=newtita["Name"].str.split(",").str.get(0)


# In[635]:


newtita["surname"]


# In[636]:


newtita


# In[640]:


newtita[newtita["Name"].str.contains("Countess")]


# In[643]:


#to find the longest name 
newtita["Name"].str.len()


# In[647]:


newtita["Name"].str.len().idxmax()


# In[651]:


newtita.loc[newtita["Name"].str.len().idxmax(),"Name"]


# In[658]:


newtita["Sex"].replace({"male":"M","female":"F"})


# In[655]:


newtita["change"]


# In[656]:


newtita


# In[ ]:




