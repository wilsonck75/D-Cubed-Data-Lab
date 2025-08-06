#!/usr/bin/env python3
"""
Calculate actual R² scores for the EM factor model to correct the markdown
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the actual data
print("📊 Loading actual EM and macro data...")
data = pd.read_csv('data/combined_em_macro_data.txt', index_col='date', parse_dates=True)

# Define the columns
em_cols = ['Brazil_EWZ', 'India_INDA', 'China_FXI', 'SouthAfrica_EZA', 'Mexico_EWW', 'Indonesia_EIDO']
macro_cols = ['USD_Index', 'Oil_Brent', 'US_10Y_Yield', 'Fed_Funds', 'VIX', 'Copper']

# Calculate returns
em_returns = data[em_cols].pct_change().dropna()
macro_returns = data[macro_cols].pct_change().dropna()

# Align the data
common_dates = em_returns.index.intersection(macro_returns.index)
em_returns_aligned = em_returns.loc[common_dates]
macro_returns_aligned = macro_returns.loc[common_dates]

# Standardize macro factors for PCA
scaler = StandardScaler()
macro_scaled = scaler.fit_transform(macro_returns_aligned)

# Apply PCA
pca = PCA(n_components=3)
macro_pca = pca.fit_transform(macro_scaled)

# Create principal components DataFrame
pc_df = pd.DataFrame(
    macro_pca,
    index=macro_returns_aligned.index,
    columns=['PC1', 'PC2', 'PC3']
)

print(f"📈 PCA Explained Variance:")
print(f"PC1: {pca.explained_variance_ratio_[0]:.1%}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.1%}")
print(f"PC3: {pca.explained_variance_ratio_[2]:.1%}")
print(f"Total: {sum(pca.explained_variance_ratio_[:3]):.1%}")

# Calculate actual R² scores for each EM ETF
print(f"\n🔍 Actual R² Scores:")
r2_results = {}

for em_etf in em_cols:
    # Get aligned data
    y = em_returns_aligned[em_etf].dropna()
    X = pc_df.loc[y.index]
    
    # Remove any remaining NaN values
    valid_mask = ~(y.isnull() | X.isnull().any(axis=1))
    y_clean = y[valid_mask]
    X_clean = X[valid_mask]
    
    if len(y_clean) > 0:
        # Fit regression model
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        
        # Calculate R²
        r2 = model.score(X_clean, y_clean)
        r2_results[em_etf] = r2
        
        print(f"{em_etf}: R² = {r2:.3f}")
    else:
        print(f"{em_etf}: No valid data")

# Sort by R² to see ranking
print(f"\n📊 R² Ranking (Highest to Lowest):")
sorted_r2 = sorted(r2_results.items(), key=lambda x: x[1], reverse=True)
for i, (etf, r2) in enumerate(sorted_r2, 1):
    print(f"{i}. {etf}: {r2:.3f}")

print(f"\nAverage R²: {np.mean(list(r2_results.values())):.3f}")
print(f"Range: {min(r2_results.values()):.3f} - {max(r2_results.values()):.3f}")
