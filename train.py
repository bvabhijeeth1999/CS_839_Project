import pandas as pd

# Define the file path
file_path = "uci-news-aggregator.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

print(df.head())
print(df.columns)

df = df.iloc[:, [1, 5]]
print(df.head())

df['story_encoded'] = pd.factorize(df.iloc[:, 1])[0]

print(df.head(50))