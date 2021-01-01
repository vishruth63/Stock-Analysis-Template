
####################################################################################
############### WELCOME TO STOCK DATA ANALYSIS AND VISUALISATION ###################
####################################################################################
"""

Author : Vishruth Khare
LinkedIn: https://www.linkedin.com/in/vishruth-khare-210b78161/

About the Project:
This project standardises the template to analyse any/multiple stock data. It also provides
a lucrative dashboard and several dynamic features that can be called upon to analyse the
data. Feel free to download, use and modify the existing codes. Constructive critism is
always welcome :)

Also, for beginners, I will explain the functionalities in depth as we encounter that
piece of code. I will also list the changes you need to do in the code in case you are not
interested in understanding the various intricacies of the code. So sit back and relax!!

"""

# Importing the dataset
"""
numpy -> scientific computation 
pandas -> Data manipulation and analysis
"""
import numpy as np
import pandas as pd

'''
NOTE : This is one of the places where you need to change the code in case your are working 
with different dataset. Just import that file in the  pd.read_...('') method. Since we 
have employed csv file here, it's always useful to export the file in the relevant format first
'''
dataset = pd.read_csv('stock.csv')
dataset.sort_index()    # Sorting because why not. Who likes unstructured data :P

# Relevant details about the dataset
print( "Dataset : ")
print(dataset.head())

print (" Length of dataset : " , len(dataset) )
print (" Number of stocks into consideration : " , (dataset.size/len(dataset)) - 1.0 )
print (" The stocks are : " , end = " ")    # Stocks under consideration
for ele in dataset.columns[1: ]:
    print(ele, end =  "   ")
print()
print()
print()
print()
####################################################################################
# So if you are a Data Scientist, mathematical correlations matter the most to you and
# Therefore we have the Statistical Summary as well
####################################################################################
print( "Statistial Summary: " )
print(dataset.describe())
print()
print()
print()
print()

####################################################################################
# Now that we have the data imported, let's dive right into the Visualisation Part
# We'll be making use of matplotlib and plotly for static and dynamic charts/graphs
####################################################################################
import matplotlib.pyplot as plt
def show_plot (df):         # This is the function we'll be using over and over. It
    df.plot()               # is a standard technique to show the charts
    plt.grid()
    plt.show()

show_plot(dataset)

####################################################################################
# What is NORMALIZATION ?
# Normalization means checking the growth of stock w.r.t the initial price of listing
# This is dont to check the multiplicity the stock has risen since it's inception
# Gives a more qualitative measure about the stock than simply looking at the max and min
####################################################################################
def normalize (df):
    x = df.copy()
    for ele in x.columns[1: ]:
        x[ele] = x[ele] / x[ele][0]
    return x

norm_dataset = normalize(dataset)
show_plot(norm_dataset)


####################################################################################
# plotly library helps in deploying these interactive charts
# Again, It's a very standard procedure for building dynamic dashboards
####################################################################################
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
def interactive_plot ( df , title ):
    fig = px.line(title = title )
    for ele in df.columns[1:]:
        fig.add_scatter(x = df['Date'], y = df[ele] , name = ele )
    fig.show()

interactive_plot(dataset, "Prices")
interactive_plot(norm_dataset, "Normalised Prices")


"""
Whoa! Congrats. You've followed through here. Impressive 
Till this point, the procedure is pretty standard and enough to comfortably visualise the
data. Beyond this point, we will get in depth of the computation that helps us draw 
connection among the data via histograms, heatmaps. This is the place where you can 
code according to your needs. A sample of max and min prices is used here but you can be
creative and compute parameters according to your needs. 
"""


# Calculate your daily returns
# Daily returns percentage has been used  ( current - prev ) / prev * 100
def daily_return (df):
     df_daily = df.copy()
     for ele in df.columns[1:]:
         for item in range(1, len(df)):
             df_daily[ele][item] = (df[ele][item] - df[ele][item-1])/(df[ele][item-1])*100
         df_daily[ele][0] = 0
     return df_daily

check = daily_return(dataset)
print(check.head())
print()
print()
print()
print()
interactive_plot(check, "Daily return")

####################################################################################
####################################################################################

def find_max_return(df):
    for ele in df.columns[1:]:
        print(ele)
        print ( "  MAX -> " , check[ele].max() , " && MIN - > " , check[ele].min() )

# find_max_return(dataset)

####################################################################################
# seaborn -> statistical graphics
####################################################################################
import seaborn as sns
cm = check.drop(columns = ['Date']).corr()
print(cm)
cm = np.array(cm)
plt.figure(figsize=(10, 10))
print(type(cm))
# ax = plt.subplot()
ax = sns.heatmap(data = cm, annot = True );

### https://stackoverflow.com/questions/26597116/seaborn-plots-not-showing-up

plt.show()
v = check.hist(figsize=(10, 10), bins = 40);
plt.show()

df_hist = check.copy()
df_hist = df_hist.drop(columns = ['Date'])
data = []
# Loop through every column
for i in df_hist.columns:
    data.append(check[i].values)

fig = ff.create_distplot(data, df_hist.columns)
fig.show()

####################################################################################
####################################################################################
####################################################################################
"""
I will be uploading and updating the codes from time to time, as and more relevant 
features pop up. Have a project in mind ? Let's discuss it on LinkedIn.  
"""