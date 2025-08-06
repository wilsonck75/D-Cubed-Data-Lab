import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('data/combined_em_macro_data.txt', index_col='date', parse_dates=True)

# Calculate returns
returns = data.pct_change().dropna()

# Split EM and macro
em_cols = ['Brazil_EWZ', 'India_INDA', 'China_FXI', 'SouthAfrica_EZA', 'Mexico_EWW', 'Indonesia_EIDO']
macro_cols = ['USD_Index', 'Oil_Brent', 'US_10Y_Yield', 'Fed_Funds', 'VIX', 'Copper']

X = returns[macro_cols]
Y = returns[em_cols]

# PCA on macro factors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Calculate R² for each EM index
results = {}
for col in em_cols:
    model = LinearRegression()
    model.fit(X_pca, Y[col])
    r2 = model.score(X_pca, Y[col])
    results[col] = r2
    print(f"{col}: {r2:.3f}")

# Sort by R²
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print(f"\nRanking:")
for i, (name, r2) in enumerate(sorted_results, 1):
    print(f"{i}. {name}: {r2:.3f}")
