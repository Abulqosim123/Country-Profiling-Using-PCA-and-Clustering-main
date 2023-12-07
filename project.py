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

Data = pd.read_csv('Country-data.csv')
df = pd.DataFrame(Data)
df.head(5)
rows, col = df.shape
print ("Dimensions of dataset: {}" . format (df.shape))
print ('Rows:', rows, '\nColumns:', col)
print(f'The data type contains:\n object --> {df.dtypes.value_counts()[2]}\n int64 --> {df.dtypes.value_counts()[1]}\n float64 --> {df.dtypes.value_counts()[0]}')
df.info()
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

df.select_dtypes(include=['object']).describe().T
unique = df.nunique()
plt.figure(figsize=(20, 6))
unique.plot(kind='bar', color=colors, hatch='//')
plt.title('Unique Elements in Each Column')
plt.ylabel('Count')
for i, v in enumerate(unique.values):
    plt.text(i, v+1, str(v), color='black', fontweight='bold', ha='center')
plt.show()
df.inflation.describe()
neg_values = df['inflation'] < 0

plt.figure(figsize=(22,4))
sns.heatmap(pd.DataFrame(neg_values.value_counts()).T, cmap=my_palette, 
            annot=True, fmt='0.0f').set_title('Number of Negative Values of Inflation', fontsize=18)
plt.show()
neg_indices = df.loc[neg_values].index
neg_indices
df.iloc[neg_indices].style.set_properties(**{'background-color': "#DBEDF3"}, subset=['inflation'])
# Missing values
plt.figure(figsize=(22,4))
sns.heatmap((df.isna().sum()).to_frame(name='').T,cmap=my_palette, annot=True,
             fmt='0.0f').set_title('Count missing values', fontsize=18)
plt.show()
# Duplicated data
df.duplicated().sum()
# Separate numerical and categorical features
num_cols = pd.DataFrame (df, columns= ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp'])
cat_cols = pd.DataFrame (df, columns= ['country'])
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
        outliers_df = outliers_df.append({'Column': col, 'Outlier_index': outliers_index, 'Outlier_values': outliers_values}, ignore_index=True)
        axs[i].plot([], [], 'ro', alpha=0.5, label=f'Outliers: {outliers.sum()}')
        axs[i].legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    return outliers_df

# outliers_df = plot_numerical_features_boxplots(data=df, columns_list=num_cols, rows=3, cols=3, title='Boxplots for Outliers')
# outliers_df
# # pairplot for noises

colors = ["#283149", "#404B69", "#DBEDF3"]

num_variables = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

sns.set_style('darkgrid')
sns.set_palette(colors)
sns.pairplot(df[num_variables])
plt.suptitle('Check for Noises', y=1.03, fontsize=25)
plt.show()
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
    skew_df = skew_df.append({'Feature': feature, 'Skewness_type': skewness, 'Skewness_value': skew}, ignore_index=True)

skew_df.style.background_gradient(cmap=cmap2).set_properties(**{'font-family': 'Segoe UI'}).hide_index()
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

sorted_df = df.sort_values('child_mort')

# Get the top and bottom 10 countries by child_mort
top_10 = sorted_df.nlargest(10, 'child_mort')['country']
bottom_10 = sorted_df.nsmallest(10, 'child_mort')['country']

# Create a boolean mask for the top and bottom 10 countries
mask_top_bottom = df['country'].isin(list(top_10) + list(bottom_10))


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

