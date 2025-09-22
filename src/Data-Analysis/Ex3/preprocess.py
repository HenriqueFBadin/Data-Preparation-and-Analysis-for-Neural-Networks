import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

# https://www.kaggle.com/code/henriquefbadin/spaceship-simple-eda/edit
# https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide#Data

print("Missing values per column:\n")
print(df.isnull().sum())

print("Unique values per column:\n")
print(df.nunique())

print("\nData types of each column:\n")
print(df.dtypes)

y = df["Transported"]
X = df.drop(columns=["Transported"])

num_cols = X.select_dtypes(include=["float64"]).columns
cat_cols = X.select_dtypes(include=["object", "bool"]).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())

for c in cat_cols:
    moda = X[c].mode(dropna=True)
    X[c] = X[c].fillna(moda.iloc[0] if not moda.empty else "Unknown")

dummies = pd.get_dummies(X[["HomePlanet", "Destination"]])
dummies = dummies.astype(int)
X_enc = X.drop(columns=["HomePlanet", "Destination"]).join(dummies)

X_enc[["Deck", "CabinNum", "Side"]] = (
    X_enc["Cabin"].astype(str).str.split("/", expand=True)
)
dataframe = X_enc.drop(columns=["Cabin"])

dummies = pd.get_dummies(dataframe[["CryoSleep", "VIP", "Side"]])
dummies = dummies.astype(int)
dataframe = dataframe.drop(columns=["CryoSleep", "VIP", "Side"]).join(dummies)

print("\nEstado intermediário do dataframe:\n")
print(dataframe.head())

dummies = pd.get_dummies(dataframe["Deck"], prefix="Deck", dummy_na=False, dtype=int)
dataframe = pd.concat([dataframe.drop(columns=["Deck"]), dummies], axis=1)

dataframe["CabinNum"] = pd.to_numeric(dataframe["CabinNum"], errors="coerce")
dataframe["CabinNum"] = dataframe["CabinNum"].fillna(dataframe["CabinNum"].median())
cont_cols = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "CabinNum",
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.histplot(dataframe["FoodCourt"], bins=30, kde=True, ax=axes[0])
axes[0].set_title("FoodCourt")

sns.histplot(dataframe["Age"], bins=30, kde=True, ax=axes[1])
axes[1].set_title("Age")

fig.suptitle(
    "Distribuições: FoodCourt e Idade Antes da Normalização",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
dataframe[cont_cols] = scaler.fit_transform(dataframe[cont_cols])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.histplot(dataframe["FoodCourt"], bins=30, kde=True, ax=axes[0])
axes[0].set_title("FoodCourt")

sns.histplot(dataframe["Age"], bins=30, kde=True, ax=axes[1])
axes[1].set_title("Age")


fig.suptitle(
    "Distribuições: FoodCourt e Idade Após a Normalização",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()
