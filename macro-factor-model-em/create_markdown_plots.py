#!/usr/bin/env python3
"""
Create example plots for the enhanced markdown portfolio post
Using real EM and macro data from the project
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory for plots
os.makedirs('output/markdown_plots', exist_ok=True)

# Load the combined dataset
print("📊 Loading EM and Macro data...")
data = pd.read_csv('data/combined_em_macro_data.txt', index_col='date', parse_dates=True)
print(f"Dataset shape: {data.shape}")

# Separate EM and macro data
em_cols = ['Brazil_EWZ', 'India_INDA', 'China_FXI', 'SouthAfrica_EZA', 'Mexico_EWW', 'Indonesia_EIDO']
macro_cols = ['USD_Index', 'Oil_Brent', 'US_10Y_Yield', 'Fed_Funds', 'VIX', 'Copper']

em_data = data[em_cols]
macro_data = data[macro_cols]

# Calculate returns
em_returns = em_data.pct_change().dropna()
macro_returns = macro_data.pct_change().dropna()

print("🎨 Creating visualization 1: EM ETF Performance Over Time...")
# Plot 1: EM ETF Performance Over Time
fig, ax = plt.subplots(figsize=(12, 8))
normalized_em = (em_data / em_data.iloc[0] * 100)
for col in em_cols:
    ax.plot(normalized_em.index, normalized_em[col], label=col.replace('_', ' '), linewidth=2)

ax.set_title('Emerging Markets ETF Performance\n(Normalized to 100 = Start Date)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Normalized Price (Base = 100)', fontsize=12)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/markdown_plots/em_etf_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("🎨 Creating visualization 2: Macro Factors Evolution...")
# Plot 2: Macro Factors Evolution (Normalized)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(macro_cols):
    normalized_macro = (macro_data[col] / macro_data[col].iloc[0] * 100)
    axes[i].plot(normalized_macro.index, normalized_macro, color=sns.color_palette("husl", 6)[i], linewidth=2)
    axes[i].set_title(f'{col.replace("_", " ")}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Normalized (Base = 100)')
    axes[i].grid(True, alpha=0.3)
    axes[i].tick_params(axis='x', rotation=45)

plt.suptitle('Macroeconomic Factors Evolution Over Time', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/markdown_plots/macro_factors_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

print("🎨 Creating visualization 3: Correlation Heatmap...")
# Plot 3: Correlation Matrix
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = em_returns.corrwith(macro_returns.mean(axis=1)).to_frame('Macro_Composite')
for macro_col in macro_cols:
    corr_matrix[macro_col] = em_returns.corrwith(macro_returns[macro_col])

corr_matrix = corr_matrix.drop('Macro_Composite', axis=1)
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
            square=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('EM ETF Returns vs Macro Factor Returns\nCorrelation Matrix', fontsize=16, fontweight='bold')
plt.ylabel('EM ETFs', fontsize=12)
plt.xlabel('Macro Factors', fontsize=12)
plt.tight_layout()
plt.savefig('output/markdown_plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("🎨 Creating visualization 4: PCA Analysis...")
# Plot 4: PCA Analysis
# Standardize the data
scaler = StandardScaler()
em_scaled = scaler.fit_transform(em_returns)

# Apply PCA
pca = PCA()
pca_components = pca.fit_transform(em_scaled)

# Create PCA plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Explained variance
ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='10% Threshold')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('PCA: Explained Variance by Component')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cumulative explained variance
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
ax2.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'o-', color='orange', linewidth=2)
ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Threshold')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('PCA: Cumulative Explained Variance')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Principal Component Analysis of EM ETF Returns', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/markdown_plots/pca_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("🎨 Creating visualization 5: Factor Model R² Scores...")
# Plot 5: Factor Model Performance (R² scores)
r2_scores = {}

for em_etf in em_cols:
    # Prepare data for regression
    y = em_returns[em_etf].dropna()
    X = macro_returns.loc[y.index]
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate R²
    r2_scores[em_etf] = r2_score(y, y_pred)

# Create R² scores plot
fig, ax = plt.subplots(figsize=(10, 6))
etf_names = [name.replace('_', ' ') for name in r2_scores.keys()]
scores = list(r2_scores.values())

bars = ax.bar(etf_names, scores, alpha=0.7, color=sns.color_palette("viridis", len(scores)))

# Add value labels on bars
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('R² Score')
ax.set_title('Factor Model Performance: R² Scores by EM ETF\n(How well macro factors explain EM returns)', 
             fontsize=14, fontweight='bold')
ax.set_ylim(0, max(scores) * 1.2)
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/markdown_plots/factor_model_r2_scores.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ All visualizations created successfully!")
print(f"📁 Plots saved to: output/markdown_plots/")

# Print summary statistics for the markdown
print("\n📊 Summary Statistics for Markdown:")
print(f"Dataset period: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
print(f"Total observations: {len(data)}")
print(f"EM ETFs: {len(em_cols)}")
print(f"Macro factors: {len(macro_cols)}")
print(f"Average R² across all ETFs: {np.mean(list(r2_scores.values())):.3f}")
print(f"Best performing model: {max(r2_scores.keys(), key=r2_scores.get)} (R² = {max(r2_scores.values()):.3f})")
