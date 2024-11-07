import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import plotly.express as px

# Load the data
df = pd.read_csv(r"archive/my-data.csv")

# region clean data
# Convert columns to numeric where necessary
#print(df.columns)
#df.columns = df.columns.str.strip()
#print("Columns after stripping spaces:", df.columns)
df["Gold Average Closing Price"] = pd.to_numeric(df["Gold Average Closing Price"], errors="coerce")
df["Oil Average Closing Price"] = pd.to_numeric(df["Oil Average Closing Price"], errors="coerce")
df["DXY Average Closing Price"] = pd.to_numeric(df["DXY Average Closing Price"], errors="coerce")

# Remove rows with missing values
df = df.dropna()


#plot

# Trends over time for Gold, Oil, and DXY
fig = px.line(df, x="Year", y=["Gold Year Close", "Oil Year Close", "DXY Year Close"],
              title="Trends of Gold, Oil, and DXY over Time")
fig.show()

# Correlation_heatmap
correlation = df[['Gold Average Closing Price', 'Oil Average Closing Price', 'DXY Average Closing Price']].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap between Gold, Oil, and DXY')
plt.show()

# Box_plot to highlight outliers
fig = px.box(df, x="Year", y="Gold Year Close", title="Gold Year Close Outliers")
fig.show()

# Scatter plot for correlations (Gold vs Oil, Gold vs DXY)
fig = px.scatter(df, x="Gold Year Close", y="Oil Year Close", title="Gold vs Oil Correlation")
fig.show()

# Statistical outlier detection (Z-score method)
z_scores = st.zscore(df[['Gold Year Close', 'Oil Year Close', 'DXY Year Close']])
outliers = (abs(z_scores) > 3).any(axis=1)
outliers_data = df[outliers]

# Highlight outliers
fig = px.scatter(df, x="Year", y="Gold Year Close", color=outliers_data.index,
                 title="Gold Year Close with Outliers Highlighted")
fig.show()


#highst & lowest value in each year

values = tuple(zip((df.groupby("Year")["Gold Year Close"].max().values),
                   (df.groupby("Year")["Gold Year Close"].min().values),
                   (df.groupby("Year")["Gold Year Close"].max().index)
                   ))

values = sorted(values, key=lambda i: i[0])
maxi, mini, years = zip(*values)

# find minimum in the data
index = mini.index(min(mini))

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(years, maxi, "g-", label="maximum price")
ax.plot(years, mini, "r-", label="minimum price")
ax.scatter(years[-1], maxi[-1], color="g", s=7, label=f"max price:{years[-1]}")
ax.scatter(years[index], mini[index], color="r", s=25, label=f"min price:{years[index]}")
ax.set_title("Gold Year Close (Max & Min)", fontsize=20)
ax.set_ylabel("Price")
ax.set_xlabel('Year')
ax.legend()
ax.grid()
plt.show()


#outlier handling and average price

df_avrage = df
average_price = np.average(df_avrage["Gold Year Close"].values)
std = np.std(df_avrage["Gold Year Close"].values)


def draw_chart(df, df1, df2, std, average_price):
    plt.scatter(df["Gold Year Close"].index, df["Gold Year Close"], color="g", s=10, label="Normal Data")
    plt.scatter(df1["Gold Year Close"].index, df1["Gold Year Close"], color="r", s=10, label="Outlier Data")
    plt.scatter(df2["Gold Year Close"].index, df2["Gold Year Close"], color="r", s=10)
    plt.plot([0, df["Gold Year Close"].index[-1]], [average_price + (2 * std), average_price + (2 * std)], color='r',
             label="std line")
    plt.plot([0, df["Gold Year Close"].index[-1]], [average_price - (2 * std), average_price - (2 * std)], color='r')
    plt.plot([0, df["Gold Year Close"].index[-1]], [average_price, average_price], color='b', label="average line")
    plt.ylabel("Price")
    plt.xlabel("Index")
    plt.legend()
    plt.title("Handle Outlier Data for Average", fontsize=16)
    plt.show()


# Handle outliers by removing those outside of average ± 2 standard deviations
while True:
    num1 = df_avrage[df_avrage["Gold Year Close"] > average_price + (2 * std)]

    if num1.empty:
        num2 = df_avrage[df_avrage["Gold Year Close"] < average_price - (2 * std)]
    else:
        num2 = num1

    if num1.empty and num2.empty:
        break

    draw_chart(df_avrage, num1, num2, std, average_price)
    df_avrage = df_avrage[df_avrage["Gold Year Close"].between(average_price - (2 * std), average_price + (2 * std))]
    average_price = np.average(df_avrage["Gold Year Close"].values)
    std = np.std(df_avrage["Gold Year Close"].values)

df_avrage = df_avrage[df_avrage["Gold Year Close"].between(average_price - (2 * std), average_price + (2 * std))]
average_price = np.average(df_avrage["Gold Year Close"].values)
std = np.std(df_avrage["Gold Year Close"].values)

plt.scatter(np.arange(len(df_avrage["Gold Year Close"])), df_avrage["Gold Year Close"], color="g", s=10)
plt.plot([0, len(df_avrage["Gold Year Close"])], [average_price, average_price], color='b', label=f"average line")
plt.plot([0, len(df_avrage["Gold Year Close"])], [average_price + (2 * std), average_price + (2 * std)], color='r',
         label=f"std line")
plt.plot([0, len(df_avrage["Gold Year Close"])], [average_price - (2 * std), average_price - (2 * std)], color='r')
plt.legend(loc="upper left")
plt.ylabel("Price")
plt.xlabel("Index")
plt.title(f"Average Gold Price\n Average price : {average_price}", fontsize=16)

plt.show()
