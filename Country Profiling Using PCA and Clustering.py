#!/usr/bin/env python
# coding: utf-8

# <a id="1"></a> 
# # <p style="padding:15px;background-color:#283149;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:500">1. Import Libraries</p> 

# In[13]:


# Data
import pandas as pd
import numpy as np
from scipy import stats

# tqdm library for progress bars
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
from termcolor import colored

# Clustering Algorithms
from sklearn.cluster import KMeans

# PCA
from sklearn.decomposition import PCA

# kneed library for finding the knee/elbow point in a plot
from kneed import KneeLocator, DataGenerator

# sklearn library
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Scaling
from sklearn.preprocessing import StandardScaler

# ignoring the warnings while executing codes
import warnings
warnings.filterwarnings('ignore')

# Color pallet and theme
import matplotlib.colors

# Define custom colors
"#283149"  # Dark blue
"#404B69"  # Medium blue
"#DBEDF3"  # Light blue
"#DBDBDB"  # Gray
"#FFFFFF"  # White

colors = ["#283149", "#404B69", "#DBEDF3", "#DBDBDB", "#FFFFFF"]
colors2 = ["#FFFFFF", "#DBDBDB", "#DBEDF3", "#404B69", "#283149"]
colors3 = ['#404B69', '#5CDB95', '#ED4C67', '#F7DC6F']

my_palette = sns.color_palette(["#283149", "#404B69", "#DBEDF3", "#DBDBDB", "#FFFFFF"])

cmap = matplotlib.colors.ListedColormap(colors)
cmap2 = matplotlib.colors.ListedColormap(colors2)

# Main Theme
sns.palplot(sns.color_palette(colors))

# Clusters Theme
sns.palplot(sns.color_palette(colors3))



# <a id="1"></a> 
# # <p style="padding:15px;background-color:#283149;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:500">2. Reading and Understanding the Dataset</p> 

# <a id="1"></a>  
# #### <p style="padding:5px;background-color:#404B69;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:200">2.1. Import Dataset</p>

# In[17]:


Data = pd.read_csv('Country-data.csv')


# In[15]:


# Preview the dataset
df = pd.DataFrame(Data)
df.head(5)


# <a id="1"></a>  
# #### <p style="padding:5px;background-color:#404B69;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:200">2.2. Discover Data</p>

# In[18]:


# View dimensions of dataset   
rows, col = df.shape
print ("Dimensions of dataset: {}" . format (df.shape))
print ('Rows:', rows, '\nColumns:', col)


# In[19]:


# Dtype
print(f'The data type contains:\n object --> {df.dtypes.value_counts()[2]}\n int64 --> {df.dtypes.value_counts()[1]}\n float64 --> {df.dtypes.value_counts()[0]}')


# In[6]:


# Information about the dataframe
df.info()


# In[20]:


# Statistical details
def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return f'color: {color}'

df_desc = df.describe(include=[np.float64, np.int64]).T.sort_values(by='std', ascending=False)
df_desc.style.background_gradient(cmap=cmap2)\
             .bar(subset=["max"])\
             .bar(subset=["mean"])\
             .applymap(color_negative_red)


# In[21]:


#Describe the column with object type
df.select_dtypes(include=['object']).describe().T


# In[22]:


# Plot the unique values
unique = df.nunique()
plt.figure(figsize=(20, 6))
unique.plot(kind='bar', color=colors, hatch='//')
plt.title('Unique Elements in Each Column')
plt.ylabel('Count')
for i, v in enumerate(unique.values):
    plt.text(i, v+1, str(v), color='black', fontweight='bold', ha='center')
plt.show()


# <a id="1"></a> 
# # <p style="padding:15px;background-color:#283149;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:500">3. Preprocessing</p> 

# In[23]:


df.inflation.describe()


# In[24]:


# Number of negative values of inflation
neg_values = df['inflation'] < 0

plt.figure(figsize=(22,4))
sns.heatmap(pd.DataFrame(neg_values.value_counts()).T, cmap=my_palette, 
            annot=True, fmt='0.0f').set_title('Number of Negative Values of Inflation', fontsize=18)
plt.show()


# In[25]:


neg_indices = df.loc[neg_values].index
neg_indices


# In[26]:


df.iloc[neg_indices].style.set_properties(**{'background-color': "#DBEDF3"}, subset=['inflation'])


# <a id="1"></a> 
# # <p style="padding:15px;background-color:#283149;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:500">4. Data Cleaning</p> 

# <a id="1"></a>  
# #### <p style="padding:5px;background-color:#404B69;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:200"> 4.1. Missing Values </p>

# In[27]:


# Missing values
plt.figure(figsize=(22,4))
sns.heatmap((df.isna().sum()).to_frame(name='').T,cmap=my_palette, annot=True,
             fmt='0.0f').set_title('Count missing values', fontsize=18)
plt.show()


# <a id="1"></a>  
# #### <p style="padding:5px;background-color:#404B69;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:200">4.2. Duplicated Data</p>

# In[28]:


# Duplicated data
df.duplicated().sum()


# <a id="1"></a>  
# #### <p style="padding:5px;background-color:#404B69;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:200">4.3. Outliers</p>

# In[29]:


# Separate numerical and categorical features
num_cols = pd.DataFrame (df, columns= ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp'])
cat_cols = pd.DataFrame (df, columns= ['country'])


# In[30]:


def plot_numerical_features_boxplots(data, columns_list, rows, cols, title):
    sns.set_style('darkgrid')
    fig, axs = plt.subplots(rows, cols, figsize=(15, 7), sharey=True)
    fig.suptitle(title, fontsize=25, y=1)
    axs = axs.flatten()
    outliers_df = pd.DataFrame(columns=['Column', 'Outlier_index', 'Outlier_values'])
    for i, col in enumerate(columns_list):
        sns.boxplot(x=data[col], color='#404B69', ax=axs[i])
        axs[i].set_title(f'{col} (skewness: {data[col].skew().round(2)})', fontsize=12)
        #----------------------------
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))
        outliers_index = data[outliers].index.tolist()
        outliers_values = data[col][outliers].tolist()
        outliers_df = outliers_df._append({'Column': col, 'Outlier_index': outliers_index, 'Outlier_values': outliers_values}, ignore_index=True)
        axs[i].plot([], [], 'ro', alpha=0.5, label=f'Outliers: {outliers.sum()}')
        axs[i].legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    return outliers_df


# In[31]:


outliers_df = plot_numerical_features_boxplots(data=df, columns_list=num_cols, rows=3, cols=3, title='Boxplots for Outliers')


# In[32]:


outliers_df


# <a id="1"></a>  
# #### <p style="padding:5px;background-color:#404B69;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:200">4.4. Noises</p>

# In[34]:


# pairplot for noises

colors = ["#283149", "#404B69", "#DBEDF3"]

num_variables = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

sns.set_style('darkgrid')
sns.set_palette(colors)
sns.pairplot(df[num_variables])
plt.suptitle('Check for Noises', y=1.03, fontsize=25)
plt.show()


# <a id="1"></a> 
# # <p style="padding:15px;background-color:#283149;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:500">5. Exploratory Data Analysis (EDA)</p> 

# <a id="1"></a>  
# #### <p style="padding:5px;background-color:#404B69;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:200"> 5.1. Univariate Analysis </p>

# In[35]:


# Check variables distribution

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))

# Define colors to use for each feature
colors = ["#283149"]

# Loop through each column and plot distribution
for i, column in enumerate(num_cols):
    # Plot histogram with density curve
    sns.distplot(df[column], color=colors[i%1], ax=axes[i//2, i%2])
    
    # Add vertical lines for mean, median, Q1 and Q3
    axes[i//2, i%2].axvline(x=df[column].median(), color='#e33434', linestyle='--', linewidth=2, label='Median')
    axes[i//2, i%2].axvline(x=df[column].quantile(0.25), color='orange', linestyle='--', linewidth=2, label='Q1')
    axes[i//2, i%2].axvline(x=df[column].quantile(0.75), color='#177ab0', linestyle='--', linewidth=2, label='Q3')
    
    # Add text box with important statistics
    median = df[column].median()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    axes[i//2, i%2].text(0.95, 0.95, 
                          'Mean: {:.2f}\nMedian: {:.2f}\nQ1: {:.2f}\nQ3: {:.2f}\nIQR: {:.2f}\nMin: {:.2f}\nMax: {:.2f}'.format(
                              df[column].mean(), median, q1, q3, iqr, df[column].min(), df[column].max()),
                          transform=axes[i//2, i%2].transAxes,
                          fontsize=10, va='top', ha='right')
    
    # Add legend
    axes[i//2, i%2].legend(loc = "upper left")
    
    # Set title of subplot
    axes[i//2, i%2].set_title('Distribution of '+ column)
    
# Add overall title and adjust spacing
fig.suptitle('Distribution of Numerical Variables', fontsize=16)
fig.tight_layout()


# In[22]:


# Check variables skewness

# create a list of features to check
features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

# create an empty DataFrame to store the results
skew_df = pd.DataFrame(columns=['Feature', 'Skewness_type'])

# loop over each feature and calculate its skewness
for feature in features:
    skew = df[feature].skew()
    
    # determine whether the skewness is positive or negative
    if skew > 0:
        skewness = 'Positively skewed'
    elif skew < 0:
        skewness = 'Negatively skewed'
    else:
        skewness = 'Symmetric'
    
    # add the results to the DataFrame
    skew_df = skew_df._append({'Feature': feature, 'Skewness_type': skewness, 'Skewness_value': skew}, ignore_index=True)
    skew_df.style.background_gradient(cmap=cmap2).set_properties(**{'font-family': 'Segoe UI'}).hide_index_()

 




# Features of each country

# Define a list of colors to use
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# Get unique values of the "country" feature
countries = df.country.unique()

# Define variable names list
var_names = ['child_mort', 'exports', 'health', 'imports', 'inflation', 'life_expec', 'total_fer']

for i, country in enumerate(countries):
    # Select data
    country_data = df[df["country"] == country]

    # Create polar scatter plot with custom color
    data = [go.Scatterpolar(
            r=[country_data[var_name].values[0] for var_name in var_names],
            theta=var_names,
            fill='toself',
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(color=colors[i % len(colors)], size=10))]

    # Define layout
    layout = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
            )
        ),
        showlegend=False,
        title=f"{country}"
    )

    # Create figure and display plot
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


# In[24]:


# Features of each country

# List of features
features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

# Loop through each feature and generate a choropleth map
for feature in features:
    fig = px.scatter_geo(df, locations='country', locationmode='country names', color=feature, color_continuous_scale='Viridis',
                         range_color=(0, df[feature].max()), title=f"Countries by {feature.replace('_', ' ').title()}",
                         labels={feature: feature.replace('_', ' ').title()}, height=700, projection='orthographic')

    # Add colorbar title
    fig.update_layout(coloraxis_colorbar=dict(title=feature.replace('_', ' ').title()))

    # Show the map
    fig.show()


# In[37]:


sorted_df = df.sort_values('child_mort')

# Get the top and bottom 10 countries by child_mort
top_10 = sorted_df.nlargest(10, 'child_mort')['country']
bottom_10 = sorted_df.nsmallest(10, 'child_mort')['country']

# Create a boolean mask for the top and bottom 10 countries
mask_top_bottom = df['country'].isin(list(top_10) + list(bottom_10))


# In[26]:


# Create the bar plot
plt.figure(figsize=(20,6))
sns.set_style('whitegrid')
ax = sns.barplot(x='country', y='child_mort', data=df[mask_top_bottom], palette='Blues', hatch='//', order=top_10)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# Add labels and title
plt.xlabel('Country')
plt.ylabel('Child Mortality Rate')
plt.title('Child Mortality Rates - Top 10 Countries')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=30)

# Display the plot
plt.show()


# In[27]:


sorted_df.tail(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['child_mort'])


# In[38]:


# Create the bar plot
plt.figure(figsize=(20,6))
sns.set_style('whitegrid')
ax = sns.barplot(x='country', y='child_mort', data=df[mask_top_bottom], palette='Blues', hatch='//', order=bottom_10)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# Add labels and title
plt.xlabel('Country')
plt.ylabel('Child Mortality Rate')
plt.title('Child Mortality Rates - Bottom 10 Countries')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=30)

# Display the plot
plt.show()


# In[29]:


sorted_df.head(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['child_mort'])


# In[39]:


print('📌The risk of death for children under 5 years of age per 1000 live births varies greatly between countries. In fact, the highest mortality country (Haiti) has a risk of child mortality that is approximately', 
      round(df['child_mort'].max() / df['child_mort'].min(), 2),
      'times higher than the lowest mortality country Iceland.')


# # In[40]:


sorted_df = df.sort_values('exports')

# Get the top and bottom 10 countries by child_mort
top_10 = sorted_df.nlargest(10, 'exports')['country']
bottom_10 = sorted_df.nsmallest(10, 'exports')['country']

# Create a boolean mask for the top and bottom 10 countries
mask_top_bottom = df['country'].isin(list(top_10) + list(bottom_10))


# # In[32]:


# Create the bar plot
plt.figure(figsize=(20,6))
sns.set_style('whitegrid')
ax = sns.barplot(x='country', y='exports', data=df[mask_top_bottom], palette='Blues', hatch='//', order=top_10)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# Add labels and title
plt.xlabel('Country')
plt.ylabel('Exports Rate')
plt.title('Exports Rates - Top 10 Countries')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=30)

# Display the plot
plt.show()


# In[33]:


sorted_df.tail(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['exports'])


# <div style="border-radius:5px;border:#283149 solid;background-color:#FFFFFF; padding:20px; font-size:15px">
# 
# 
# - Interestingly, **Singapore** has managed to become a leader in exports. Similarly, countries like **Luxembourg** and **Malta** have also achieved high export volumes by leveraging their unique strengths.

# In[34]:


# Create the bar plot
plt.figure(figsize=(20,6))
sns.set_style('whitegrid')
ax = sns.barplot(x='country', y='exports', data=df[mask_top_bottom], palette='Blues', hatch='//', order=bottom_10)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# Add labels and title
plt.xlabel('Country')
plt.ylabel('Exports Rate')
plt.title('Exports Rates - Bottom 10 Countries')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=30)

# Display the plot
plt.show()


# # In[35]:


# sorted_df.head(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['exports'])


# # In[36]:


# print('📌The export volumes of Singapore is', 
#       round(df['exports'].max() / df['exports'].min(), 2),
#       'times higher than that of Myanmar.')


# # In[41]:


# sorted_df = df.sort_values('health')

# # Get the top and bottom 10 countries by child_mort
# top_10 = sorted_df.nlargest(10, 'health')['country']
# bottom_10 = sorted_df.nsmallest(10, 'health')['country']

# # Create a boolean mask for the top and bottom 10 countries
# mask_top_bottom = df['country'].isin(list(top_10) + list(bottom_10))


# # In[38]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='health', data=df[mask_top_bottom], palette='Blues', hatch='//', order=top_10)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('Health Rate')
# plt.title('Health Rates - Top 10 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[39]:


# sorted_df.tail(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['health'])


# # In[40]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='health', data=df[mask_top_bottom], palette='Blues', hatch='//', order=bottom_10)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('Health Rate')
# plt.title('Health Rates - Bottom 10 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[41]:


# sorted_df.head(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['health'])


# # In[42]:


# print('📌The United States spends', 
#       round(df['health'].max() / df['health'].min(), 2),
#       'times more on healthcare than the country of Qatar.')


# # In[43]:


# sorted_df = df.sort_values('imports')

# # Get the top and bottom 10 countries by child_mort
# top_10 = sorted_df.nlargest(10, 'imports')['country']
# bottom_10 = sorted_df.nsmallest(10, 'imports')['country']

# # Create a boolean mask for the top and bottom 10 countries
# mask_top_bottom = df['country'].isin(list(top_10) + list(bottom_10))


# # In[44]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='imports', data=df[mask_top_bottom], palette='Blues', hatch='//', order=top_10)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('Imports Rate')
# plt.title('Imports Rates - Top 10 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[45]:


# sorted_df.tail(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['imports'])


# # In[46]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='imports', data=df[mask_top_bottom], palette='Blues', hatch='//', order=bottom_10)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('Imports Rate')
# plt.title('Imports Rates - Bottom 10 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[47]:


# sorted_df.head(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['imports'])


# # In[48]:


# print('📌The import volumes of Singapore is', 
#       round(df['imports'].max() / df['imports'].min(), 2),
#       'times higher than that of Myanmar.')


# # In[49]:


# sorted_df = df.sort_values('income')

# # Get the top and bottom 10 countries by child_mort
# top_10 = sorted_df.nlargest(10, 'income')['country']
# bottom_10 = sorted_df.nsmallest(10, 'income')['country']

# # Create a boolean mask for the top and bottom 10 countries
# mask_top_bottom = df['country'].isin(list(top_10) + list(bottom_10))


# # In[50]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='income', data=df[mask_top_bottom], palette='Blues', hatch='//', order=top_10)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('Income Rate')
# plt.title('Income Rates - Top 10 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[51]:


# sorted_df.tail(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['income'])


# # In[52]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='income', data=df[mask_top_bottom], palette='Blues', hatch='//', order=bottom_10)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('Income Rate')
# plt.title('Income Rates - Bottom 10 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[53]:


# sorted_df.head(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['income'])


# # In[54]:


# print('📌Net annual income per person in Qatar is', 
#       round(df['income'].max() / df['income'].min(), 2),
#       'times higher than that of Congo, Dem. Rep.')


# # In[55]:


# neg_inflation = df.iloc[neg_indices]
# neg_inflation = neg_inflation.sort_values('inflation')


# # In[56]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='inflation', data=neg_inflation, palette='Blues', hatch='//')
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('Inflation Rate')
# plt.title('Countries with deflation or negative inflation Rates')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[57]:


# sorted_df = df.sort_values('inflation')

# # Get the top and bottom 10 countries by child_mort
# top_10 = sorted_df.nlargest(10, 'inflation')['country']
# bottom_20 = sorted_df.nsmallest(20, 'inflation')['country']

# # Create a boolean mask for the top and bottom 10 countries
# mask_top_bottom = df['country'].isin(list(top_10) + list(bottom_20))


# # In[58]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='inflation', data=df[mask_top_bottom], palette='Blues', hatch='//', order=top_10)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('Inflation Rate')
# plt.title('Inflation Rates - Top 10 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[59]:


# sorted_df.tail(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['inflation'])


# # In[60]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='inflation', data=df[mask_top_bottom], palette='Blues', hatch='//', order=bottom_20)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add a horizontal and vertical line
# ax.axhline(y=0, color='black', linewidth=1)
# ax.axvline(x=7.5, color='black', linestyle='--', linewidth=1)

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('Inflation Rate')
# plt.title('Inflation Rates - Bottom 20 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[61]:


# sorted_df.head(20).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['inflation'])


# # In[62]:


# sorted_df = df.sort_values('life_expec')

# # Get the top and bottom 10 countries by child_mort
# top_10 = sorted_df.nlargest(10, 'life_expec')['country']
# bottom_10 = sorted_df.nsmallest(10, 'life_expec')['country']

# # Create a boolean mask for the top and bottom 10 countries
# mask_top_bottom = df['country'].isin(list(top_10) + list(bottom_10))


# # In[63]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='life_expec', data=df[mask_top_bottom], palette='Blues', hatch='//', order=top_10)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('Life expectancy Rate')
# plt.title('Life expectancy Rates - Top 10 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[64]:


# sorted_df.tail(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['life_expec'])


# # In[65]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='life_expec', data=df[mask_top_bottom], palette='Blues', hatch='//', order=bottom_10)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('Life expectancy Rate')
# plt.title('Life expectancy Rates - Bottom 10 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[66]:


# sorted_df.head(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['life_expec'])


# # In[67]:


# print('📌Life expectancy in Japan is', 
#       round(df['life_expec'].max() / df['life_expec'].min(), 2),
#       'times higher than that of Haiti.')


# # In[68]:


# sorted_df = df.sort_values('total_fer')

# # Get the top and bottom 10 countries by child_mort
# top_10 = sorted_df.nlargest(10, 'total_fer')['country']
# bottom_10 = sorted_df.nsmallest(10, 'total_fer')['country']

# # Create a boolean mask for the top and bottom 10 countries
# mask_top_bottom = df['country'].isin(list(top_10) + list(bottom_10))


# # In[69]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='total_fer', data=df[mask_top_bottom], palette='Blues', hatch='//', order=top_10)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('Total fertility Rate')
# plt.title('Total fertility Rates - Top 10 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[70]:


# sorted_df.tail(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['total_fer'])


# # In[71]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='total_fer', data=df[mask_top_bottom], palette='Blues', hatch='//', order=bottom_10)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('Total fertility Rate')
# plt.title('Total fertility Rates - Bottom 10 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[72]:


# sorted_df.head(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['total_fer'])


# # In[73]:


# print('📌TFR in Niger is', 
#       round(df['total_fer'].max() / df['total_fer'].min(), 2),
#       'times higher than that of Singapore.')


# # In[74]:


# sorted_df = df.sort_values('gdpp')

# # Get the top and bottom 10 countries by child_mort
# top_10 = sorted_df.nlargest(10, 'gdpp')['country']
# bottom_10 = sorted_df.nsmallest(10, 'gdpp')['country']

# # Create a boolean mask for the top and bottom 10 countries
# mask_top_bottom = df['country'].isin(list(top_10) + list(bottom_10))


# # In[75]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='gdpp', data=df[mask_top_bottom], palette='Blues', hatch='//', order=top_10)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('GDP Rate')
# plt.title('GDP Rates - Top 10 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[76]:


# sorted_df.tail(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['gdpp'])


# # In[77]:


# # Create the bar plot
# plt.figure(figsize=(20,6))
# sns.set_style('whitegrid')
# ax = sns.barplot(x='country', y='gdpp', data=df[mask_top_bottom], palette='Blues', hatch='//', order=bottom_10)
# for p in ax.patches:
#     ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# # Add labels and title
# plt.xlabel('Country')
# plt.ylabel('GDP Rate')
# plt.title('GDP Rates - Bottom 10 Countries')

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=30)

# # Display the plot
# plt.show()


# # In[78]:


# sorted_df.head(10).style.set_properties(**{'background-color': '#DBEDF3'}, subset=['gdpp'])


# # In[79]:


# print("📌Luxembourg's GDP rate is", 
#       round(df['gdpp'].max() / df['gdpp'].min(), 2),
#       'times higher than that of Burundi.')


# # <a id="1"></a>  
# # #### <p style="padding:5px;background-color:#404B69;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:200"> 5.2. Multivariate Analysis </p>

# # In[45]:


# # Multivariate relationships between numeric variables
# colors = ["#283149", "#404B69", "#DBEDF3"]

# num_variables = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

# sns.set_style('darkgrid')
# sns.set_palette(colors)
# sns.pairplot(df[num_variables])
# plt.suptitle('Multivariate Relationships\n Between Numeric Variables', y=1.03, fontsize=25)
# plt.show()


# # In[46]:


# # Correlation between the features
# fig=plt.gcf()
# fig.set_size_inches(18, 12)
# plt.title('Correlation Between The Features')
# a = sns.heatmap(df.corr(), annot = True, cmap =cmap2, fmt='.2f', linewidths=0.2)
# a.set_xticklabels(a.get_xticklabels(), rotation=90)
# a.set_yticklabels(a.get_yticklabels(), rotation=30)
# plt.show()


# # In[ ]:


# # This function creates a correlation matrix to identify linear correlations between features and their correlation coefficient.
# def corr_matrix(data):
#     corr_matrix = data.corr()
    
#     for var in corr_matrix.columns:
#         # Keep only the correlation values that are greater than 0.7 and less than 1 in absolute value
#         corr_matrix[var] = corr_matrix[var].apply(lambda x: x if abs(x) > 0.7 and abs(x) < 1 else '')
    
#     return corr_matrix


# # In[ ]:


# corr_matrix(df)


# # In[ ]:


# sns.set_style('darkgrid')
# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 14))

# # plot 1
# sns.regplot(x='child_mort', y='life_expec', data=df, ax=axes[0][0])
# r, p = stats.pearsonr(df['child_mort'], df['life_expec'])
# axes[0][0].set_title('Child Mortality vs. Life Expectancy\nr = {:.2f}'.format(r), fontsize=12, fontweight='bold')
# axes[0][0].set_xlabel('Child Mortality')
# axes[0][0].set_ylabel('Life Expectancy')

# # plot 2
# point = df[(df['child_mort'] > 200) & (df['total_fer'] < 4)]
# sns.regplot(x='child_mort', y='total_fer', data=df, ax=axes[0][1])
# sns.scatterplot(x='child_mort', y='total_fer', data=point, color='red', ax=axes[0][1])
# axes[0][1].annotate('Haiti', xy=(point['child_mort'], point['total_fer']), xytext=(-10, 10), textcoords='offset points', fontsize=12, color='red')
# r, p = stats.pearsonr(df['child_mort'], df['total_fer'])
# axes[0][1].set_title('Child Mortality vs. Total Fertility\nr = {:.2f}'.format(r), fontsize=12, fontweight='bold')
# axes[0][1].set_xlabel('Child Mortality')
# axes[0][1].set_ylabel('Total Fertility')

# # plot 3
# point = df[(df['life_expec'] < 35) & (df['total_fer'] > 3)]
# sns.regplot(x='life_expec', y='total_fer', data=df, ax=axes[1][0])
# sns.scatterplot(x='life_expec', y='total_fer', data=point, color='red', ax=axes[1][0])
# axes[1][0].annotate('Haiti', xy=(point['total_fer'], point['life_expec']), xytext=(10, 10), textcoords='offset points', fontsize=12, color='red')
# r, p = stats.pearsonr(df['life_expec'], df['total_fer'])
# axes[1][0].set_title('Life Expectancy vs. Total Fertility\nr = {:.2f}'.format(r), fontsize=12, fontweight='bold')
# axes[1][0].set_xlabel('Life Expectancy')
# axes[1][0].set_ylabel('Total Fertility')

# # plot 4
# sns.regplot(x='exports', y='imports', data=df, ax=axes[1][1])
# r, p = stats.pearsonr(df['exports'], df['imports'])
# axes[1][1].set_title('Exports vs. Imports\nr = {:.2f}'.format(r), fontsize=12, fontweight='bold')
# axes[1][1].set_xlabel('Exports')
# axes[1][1].set_ylabel('Imports')

# # plot 5
# sns.regplot(x='exports', y='imports', data=df, ax=axes[2][0])
# r, p = stats.pearsonr(df['gdpp'], df['income'])
# axes[2][0].set_title('GDP vs. Income\nr = {:.2f}'.format(r), fontsize=12, fontweight='bold')
# axes[2][0].set_xlabel('GDP')
# axes[2][0].set_ylabel('Income')

# # Adjust spacing between subplots
# fig.tight_layout()

# plt.show()


# # <a id="1"></a> 
# # # <p style="padding:15px;background-color:#283149;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:500">6. Model Building</p> 

# # <a id="1"></a>  
# # #### <p style="padding:5px;background-color:#404B69;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:200"> 6.1. Feature Scaling </p>

# # In[ ]:


# # Set the index to 'country'
# df.set_index(keys='country', drop=True, inplace=True)

# # Scale the features using StandardScaler
# scaler = StandardScaler()

# # Fit the scaler to the data and transform it
# scaled_cols = scaler.fit_transform(df)

# # Create a DataFrame from the scaled features
# scaled_cols = pd.DataFrame(scaled_cols, columns=df.columns, index=df.index)

# # Show the first few rows of the scaled data
# scaled_cols.head()


# # <a id="1"></a>  
# # #### <p style="padding:5px;background-color:#404B69;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:200"> 6.2. Principal Component Analysis (PCA) </p>

# # In[ ]:


# # Create PCA object and fit the scaled data
# pca = PCA()
# pca.fit(scaled_cols)

# # Transform the data to its principal components
# X_pca = pca.transform(scaled_cols)


# # In[ ]:


# # Get the number of principal components
# num_components = pca.n_components_

# # Print the number of principal components
# print(f"Total number of principal components = {num_components}")


# # In[ ]:


# # Print the explained variance of each principal component
# explained_variances = pca.explained_variance_
# print("Explained variance of each principal component:", explained_variances)

# # Print the total variance explained by all the principal components
# total_variance = explained_variances.sum()
# print("Total variance explained by all principal components:", total_variance)


# # In[ ]:


# # Print the explained variance ratio of each principal component
# explained_var_ratio = pca.explained_variance_ratio_
# print("Explained variance ratio of each principal component:", explained_var_ratio)

# # Print the total variance explained by all the principal components
# total_var_ratio = explained_var_ratio.sum()
# print("Total variance ratio explained by all principal components:", total_var_ratio)


# # Let’s visualize the percentage of variances captured by each dimension.

# # In[ ]:


# # Calculate percentage variation
# per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
# labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]


# # In[ ]:


# # Variance explained by each dimension
# fig, ax = plt.subplots(figsize=(12, 4))
# bars = ax.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels, color=colors, edgecolor='black', hatch='//')
# plt.ylabel('Percentage of Explained Variance')
# plt.xlabel('Principal Component')
# for i, bar in enumerate(bars):
#     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
#             f'{per_var[i]}%', ha='center', va='bottom', color='black', fontweight='bold')
# plt.show()


# # In[ ]:


# # plot pca
# fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

# # plot PCA1 vs PCA2
# axs[0].scatter(X_pca[:,0], X_pca[:,1])
# axs[0].set_title('PC1 vs. PC2', fontweight='bold')
# axs[0].set_xlabel('PC1 - {0:.1f}%'.format(per_var[0]*100))
# axs[0].set_ylabel('PC2 - {0:.1f}%'.format(per_var[1]*100))

# # plot PCA1 vs PCA3
# axs[1].scatter(X_pca[:,0], X_pca[:,2])
# axs[1].set_title('PC1 vs. PC3', fontweight='bold')
# axs[1].set_xlabel('PC1 - {0:.1f}%'.format(per_var[0]*100))
# axs[1].set_ylabel('PC3 - {0:.1f}%'.format(per_var[2]*100))

# # adjust the layout and show the figure
# plt.tight_layout()
# plt.show()


# # In[ ]:


# #Principal Component Data Decomposition
# colnames = list(df.columns)
# pca_df = pd.DataFrame({
#     'Features':colnames,
#     'PC1':pca.components_[0], 'PC2':pca.components_[1], 'PC3':pca.components_[2], 'PC4':pca.components_[3],
#     'PC5':pca.components_[4], 'PC6':pca.components_[5], 'PC7':pca.components_[6], 'PC8':pca.components_[7], 
#     'PC9':pca.components_[8]})

# pca_df


# # In[ ]:


# fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# sns.scatterplot(pca_df.PC1, pca_df.PC2, hue=pca_df.Features, marker='o', s=200, ax=axs[0])
# axs[0].set_title('PC1 vs. PC2', fontweight='bold', fontsize=15)
# axs[0].set_xlabel('PC1', fontsize=12)
# axs[0].set_ylabel('PC2', fontsize=12)

# sns.scatterplot(pca_df.PC1, pca_df.PC3, hue=pca_df.Features, marker='o', s=200, ax=axs[1])
# axs[1].set_title('PC1 vs. PC3', fontweight='bold', fontsize=15)
# axs[1].set_xlabel('PC1', fontsize=12)
# axs[1].set_ylabel('PC3', fontsize=12)

# plt.show()


# # In[ ]:


# # Create a list of integers from 1 to the number of components
# components = list(range(1, len(pca.explained_variance_ratio_) + 1))

# # Plot the cumulative explained variance ratio as a step function
# plt.step(components, np.cumsum(pca.explained_variance_ratio_), where='mid', label='Cumulative')

# # Plot the explained variance ratio for each individual component as a line
# plt.plot(components, pca.explained_variance_ratio_, marker='o', label='Individual')

# # Add horizontal line between 0.7-0.95 (optimum range)
# plt.axhline(y=0.95, color='gray', linestyle='--')
# plt.axhline(y=0.7, color='gray', linestyle='--')

# # Add axis labels and title
# plt.xlabel('Principal Component')
# plt.ylabel('Explained Variance Ratio')

# # Add a legend
# plt.legend()

# plt.show()


# # In[ ]:


# # Calculate the cumulative sum of explained variance ratios
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# # Find the number of components required to explain 90% of the variance
# n_components = np.argmax(cumulative_variance >= 0.9) + 1

# # Print the result
# print(f'{n_components} principal components explain 90% of the variance.')


# # In[ ]:


# # Initialize PCA with 5 components
# pca = PCA(n_components=n_components)

# # Fit and transform the scaled data
# pca_ = pca.fit_transform(scaled_cols)

# # Create a DataFrame from the transformed data
# pca_df = pd.DataFrame(pca_, columns=["PC1", "PC2", "PC3", "PC4", "PC5"])
# print('\nFinal PCA:')
# pca_df


# # <a id="1"></a>  
# # #### <p style="padding:5px;background-color:#404B69;margin:0;color:#DBDBDB;font-family:newtimeroman;font-size:150%;text-align:center;border-radius: 5px 5px;overflow:hidden;font-weight:200"> 6.3. k-means Clustering
# #  </p>

# # In[ ]:


# # Set the parameters for the KMeans algorithm
# kmeans_params = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 1,
#                  "tol": 1e-4, "algorithm": "auto", "verbose": 0}


# # In[47]:


# # Perform KMeans clustering for different values of k and store the inertia values
# inertia_values = []
# for num_clusters in range(1, 11):
#     kmeans = KMeans(n_clusters=num_clusters, **kmeans_params)
#     kmeans.fit(pca_df)
#     inertia_values.append(kmeans.inertia_)


# # In[48]:


# def elbow_optimizer(inertia_values, name):
#     """ Find optimom k for clustering algorithm
#         inertias (list): list that has inertia for each selected k
#         name (string): name of clustering algorithm
#     """

#     kl = KneeLocator(range(1,11), inertia_values, curve='convex', direction="decreasing")
#     plt.style.use("fivethirtyeight")
#     sns.lineplot(x=range(1,11), y=inertia_values, color=colors[1], linewidth=3)
#     plt.xticks(range(1,11))
#     plt.xlabel("Number of Clusters", labelpad=20, fontsize=12)
#     plt.ylabel("Inertia", labelpad=20, fontsize=12)
#     plt.title(f"Elbow Method for {name}", y=1.09, fontsize=14)
#     plt.axvline(x=kl.elbow, color= 'grey', label='Elbow', ls='--', linewidth=3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# # In[102]:


# # Implement elbow_optimizer function for KMeans
# elbow_optimizer(inertia_values, 'Kmeans')


# # In[103]:


# # Calculate silhouette coefficient and calinski harabasz coefficient
# silhouette_coef = []
# for k in range(2,11):
#     kmeans = KMeans(n_clusters=k, **kmeans_params)
#     kmeans.fit(pca_df)
#     score = silhouette_score(pca_df, kmeans.labels_)
#     silhouette_coef.append(score)

# calinski_harabasz_coef = []
# for k in range(2,11):
#     kmeans = KMeans(n_clusters=k, **kmeans_params)
#     kmeans.fit(pca_df)
#     score = calinski_harabasz_score(pca_df, kmeans.labels_)
#     calinski_harabasz_coef.append(score)


# # In[104]:


# # Create a dataframe to store the scores
# scores_kmeans = pd.DataFrame({'k': range(2,11),
# 'Silhouette Score': silhouette_coef,
# 'Calinski-Harabasz Score': calinski_harabasz_coef})

# # Find the best k for each score
# best_k_silhouette = scores_kmeans.loc[scores_kmeans['Silhouette Score'].idxmax(), 'k']
# best_k_calinski_harabaz = scores_kmeans.loc[scores_kmeans['Calinski-Harabasz Score'].idxmax(), 'k']


# # In[105]:


# print("\nScores for Different Numbers of Clusters:")
# scores_kmeans.style.background_gradient(cmap=cmap2).set_properties(**{'font-family': 'Segoe UI'}).hide_index()


# # In[106]:


# def plot_evaluation(sh_score, ch_score, name, x=range(2,11)):
#     """
#     for draw evaluation plot include silhouette_score and calinski_harabasz_score.
#         sh_score(list): include silhouette_score of models
#         ch_score(list): include calinski_harabasz_score of models
#         name(string): name of clustering algorithm
#         x(list): has range of number for x axis
#     """
    
#     fig, ax = plt.subplots(1,2,figsize=(15,7), dpi=100)
#     ax[0].plot(x, sh_score, color=colors[1], marker='o', ms=9, mfc=colors[-1])
#     ax[1].plot(x, ch_score, color=colors[1], marker='o', ms=9, mfc=colors[-1])
#     ax[0].set_xlabel("Number of Clusters", labelpad=20)
#     ax[0].set_ylabel("Silhouette Coefficient", labelpad=20)
#     ax[1].set_xlabel("Number of Clusters", labelpad=20)
#     ax[1].set_ylabel("calinski Harabasz Coefficient", labelpad=20)
#     plt.suptitle(f'Evaluate {name} Clustering',y=0.9)
#     plt.tight_layout(pad=3)
#     plt.show()


# # In[107]:


# # Visualize plots of silhouette and calinski harabasz scores for kmeans models
# plot_evaluation(silhouette_coef, calinski_harabasz_coef, 'Kmeans')


# # In[108]:


# # Best k for each coef
# print(f"\n▪️Best k for Silhouette Coefficient: {best_k_silhouette}")
# print(f"▪️Best k for Calinski-Harabasz Coefficient: {best_k_calinski_harabaz}")


# # In[109]:


# # Using hybrid score to find the best k
# def select_k(pca_df):
#     silhouette_scores = []
#     calinski_scores = []
#     k_values = range(2, 11)

#     for k in k_values:
#         # Perform clustering using k-means algorithm
#         kmeans = KMeans(n_clusters=k, **kmeans_params)
#         labels = kmeans.fit_predict(pca_df)

#         # Calculate Silhouette Score
#         silhouette_scores.append(silhouette_score(pca_df, labels))

#         # Calculate Calinski-Harabasz Score
#         calinski_scores.append(calinski_harabasz_score(pca_df, labels))

#     hybrid_scores = np.sqrt(np.multiply(silhouette_scores, calinski_scores))
#     best_k = k_values[np.argmax(hybrid_scores)]
#     return best_k


# # In[110]:


# best_k = select_k(pca_df)
# print("\n✅Best value of k:", best_k)


# # In[111]:


# # Implement kmeans clustering with n_clusters=4
# kmeans = KMeans(n_clusters=4, **kmeans_params).fit(pca_df)

# # Store result of kmeans
# pred = kmeans.labels_

# # Get the coordinates of the centroids of the three clusters
# centroids = kmeans.cluster_centers_

# # Convert the centroids array to a DataFrame and assign column names
# centroids = pd.DataFrame(centroids, columns = pca_df.columns)
# centroids


# # In[112]:


# # Calculate scores
# silhouette = silhouette_score(pca_df, pred)
# calinski = calinski_harabasz_score(pca_df, pred)

# # Create a DataFrame with the scores
# km_scores = pd.DataFrame({'Silhouette Score': [silhouette], 'Calinski-Harabasz Score': [calinski]}, index=['k-means'])
# km_scores


# # In[113]:


# km_clustered_df = pd.concat([pca_df, pd.DataFrame(pred, columns=['k-means Cluster'])], axis = 1)
# km_clustered_df


# # In[114]:


# # create subplots with 1 row and 2 columns
# fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

# # plot on first axis
# axs[0].set_title('PC1 vs. PC2', fontweight='bold', fontsize=14)
# axs[0].set_xlabel('PC1', fontsize=12)
# axs[0].set_ylabel('PC2', fontsize=12)
# axs[0].tick_params(axis='both', which='major', labelsize=8)
# sns.scatterplot(x='PC1', y='PC2', hue='k-means Cluster', data=km_clustered_df, ax=axs[0], palette=colors3, legend=False)

# # plot on second axis
# axs[1].set_title('PC1 vs. PC3', fontweight='bold', fontsize=14)
# axs[1].set_xlabel('PC1', fontsize=12)
# axs[1].set_ylabel('PC3', fontsize=12)
# axs[1].tick_params(axis='both', which='major', labelsize=8)
# sns.scatterplot(x='PC1', y='PC3', hue='k-means Cluster', data=km_clustered_df, ax=axs[1], palette=colors3)
# axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

# plt.show()


# # In[115]:


# # Addition of cluster column to the dataframe
# kmeans_clustered_df = df.copy()
# kmeans_clustered_df ['k-means Cluster'] = pred
# kmeans_clustered_df


# # In[116]:


# # Size of the clusters
# km_clusts_sizes = kmeans_clustered_df.groupby('k-means Cluster').size().to_frame()
# km_clusts_sizes.columns = ["Cluster Size"]
# km_clusts_sizes


# # In[117]:


# # Distribution of Data Points by Cluster
# counts = kmeans_clustered_df['k-means Cluster'].value_counts()
# labels = counts.index.tolist()
# sizes = counts.tolist()

# # Calculate percentages and counts
# total = sum(sizes)
# percentages = [(size / total) * 100 for size in sizes]
# counts = [f'{size}\n({percentages[i]:.1f}%)' for i, size in enumerate(sizes)]

# # Create the bar plot
# plt.figure(figsize=(12, 4))
# bars = plt.bar(labels, sizes, color=my_palette, edgecolor='black', linewidth=1, hatch='//')
# plt.xticks(range(len(labels)), [str(i) for i in range(len(labels))])

# # Add counts to bars
# for i, bar in enumerate(bars):
#     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
#              counts[i], ha='center', va='bottom', color='black', fontsize=12)

# # Set the axis labels and title
# plt.xlabel('Cluster', fontsize= 12)
# plt.ylabel('Count', fontsize= 12)
# plt.title('Distribution of Data Points by Cluster\n(k-means Clustering)', fontsize= 16)
# plt.show()


# # <div style="border-radius:5px;border:#283149 solid;background-color:#FFFFFF; padding:20px; font-size:15px">
# # 
# # **📝Explenations:**
# # 
# # The countries can be divided into four clusters based on certain characteristics. 
# # - The largest cluster (3), with 87 countries, covers over 50% of the total number of countries. 
# # - The second largest cluster (2), with 47 countries, covers approximately 28% of the total. 
# # - The third cluster largest (0) covers 18% of the overall countries, comprising 30 countries. 
# # - Finally, the smallest cluster (1) consists of only 3 countries, representing a mere 1.8% of the total count.

# # In[118]:


# # Cluster 1
# kmeans_clustered_df[kmeans_clustered_df['k-means Cluster'] == 0][:10]


# # In[119]:


# # Cluster 2
# kmeans_clustered_df[kmeans_clustered_df['k-means Cluster'] == 1][:10]


# # In[120]:


# # Cluster 3
# kmeans_clustered_df[kmeans_clustered_df['k-means Cluster'] == 2][:10]


# # In[121]:


# # Cluster 4
# kmeans_clustered_df[kmeans_clustered_df['k-means Cluster'] == 3][:10]


# # In[122]:


# # Summarizing the average values of each feature within each cluster
# avg_df = kmeans_clustered_df.groupby(['k-means Cluster'], as_index=False).mean()
# avg_df.style.background_gradient(cmap=cmap2).set_properties(**{'font-family': 'Segoe UI'}).hide_index()


# # In[123]:


# sns.set_style('darkgrid')
# sns.pairplot(kmeans_clustered_df, hue='k-means Cluster', palette=colors3)
# plt.show()


# # In[124]:


# # Create subplots with 3 rows and 3 columns, and set the figure size
# fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(25,20), sharex=False, sharey=False)

# # Add padding between the subplots
# fig.tight_layout(pad=3.5)

# # Loop through each feature in the data frame
# for i, feature in enumerate(kmeans_clustered_df.columns[:-1]):
#     # Create a box plot for the current feature
#     sns.boxplot(x='k-means Cluster', y=feature, data=kmeans_clustered_df, palette=colors3, ax=ax[i//3, i%3])
#     ax[i//3, i%3].set_title('{} vs. Cluster'.format(feature), fontweight='bold', fontsize=20)
#     ax[i//3, i%3].set_xlabel('Cluster', fontsize=20)
#     ax[i//3, i%3].set_ylabel(feature, fontsize=20)

# plt.show()


# # In[125]:


# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 11))

# for i, col in enumerate(['child_mort', 'income', 'life_expec', 'total_fer']):
#     sns.barplot(x='k-means Cluster', y=col, data=avg_df, ax=axes[i//2, i%2], palette=colors3)
#     axes[i//2, i%2].set_xlabel('Cluster')
#     axes[i//2, i%2].set_ylabel(col)

# plt.suptitle('Features vs. Clusters', fontsize=20)
# fig.tight_layout(pad=2.0)
# plt.show()


# # In[126]:


# kmeans_clustered_df = kmeans_clustered_df.reset_index()
# kmeans_clustered_df.loc[kmeans_clustered_df['k-means Cluster'] == 0, 'k-means Cluster'] = 'No Need for Financial Aid\n(Maybe in terms of some aspects)'
# kmeans_clustered_df.loc[kmeans_clustered_df['k-means Cluster'] == 1, 'k-means Cluster'] = 'No Need for Financial Aid'
# kmeans_clustered_df.loc[kmeans_clustered_df['k-means Cluster'] == 2, 'k-means Cluster'] = 'Need Financial Aid'
# kmeans_clustered_df.loc[kmeans_clustered_df['k-means Cluster'] == 3, 'k-means Cluster'] = 'Might Need Financial Aid'
# kmeans_clustered_df.head(5)


# # In[127]:


# # Create a choropleth map of k-means clusters
# world_help_clusters = px.choropleth(
#     data_frame=kmeans_clustered_df,
#     locations='country', 
#     locationmode='country names', 
#     color='k-means Cluster',
#     color_discrete_map={
#         'No Need for Financial Aid\n(Maybe in terms of some aspects)': '#404B69',
#         'No Need for Financial Aid': '#5CDB95', 
#         'Need Financial Aid': '#ED4C67', 
#         'Might Need Financial Aid': '#F7DC6F'
#     }, 
#     title='Worldwide Financial Aid Needs and Priorities',
#     labels={'k-means Cluster': 'Labels'}
# )

# world_help_clusters.update_geos(
#     fitbounds="locations",
#     projection=dict(type='natural earth')
# )
# world_help_clusters.update_layout(
#     legend_title_text='Labels', 
#     legend_title_side='top', 
#     title_pad_l=260, 
#     title_y=0.86
# )

# # Show the figure
# world_help_clusters.show()

