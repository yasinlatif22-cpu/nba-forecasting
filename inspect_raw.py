import pandas as pd

df = pd.read_csv("data/raw/team_advanced_2010_2025.csv")

print(df.shape)
print(df.columns.tolist())
print()
print(df.groupby("SEASON").agg(
    teams=("TEAM_ID", "nunique"),
    min_gp=("GP", "min"),
    max_gp=("GP", "max"),
))
print()
cols = ["SEASON", "TEAM_ID", "TEAM_NAME", "GP", "W", "L",
        "W_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]
print(df[cols].head(10))
