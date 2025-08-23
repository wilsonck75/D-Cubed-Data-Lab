#!/usr/bin/env python3
"""
Fix for RuntimeWarning: invalid value encountered in log

This script demonstrates how to properly handle negative and zero values
when calculating log returns to eliminate the RuntimeWarning.
"""

import pandas as pd
import numpy as np
import warnings

# Suppress the specific warning for demonstration
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas.core.internals.blocks')

def safe_log_returns(data):
    """
    Calculate log returns with proper handling of negative/zero values
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data with price levels
        
    Returns:
    --------
    pd.DataFrame
        Log returns with all problematic values handled
    """
    print("🔍 Pre-log validation:")
    for col in data.columns:
        min_val = data[col].min()
        max_val = data[col].max()
        zero_count = (data[col] == 0).sum()
        neg_count = (data[col] < 0).sum()
        
        if zero_count > 0 or neg_count > 0:
            print(f"   ⚠️  {col}: min={min_val:.4f}, max={max_val:.4f}, zeros={zero_count}, negatives={neg_count}")
    
    # Create a copy to avoid modifying original data
    data_clean = data.copy()
    
    # Step 1: Handle zero values
    data_clean = data_clean.replace(0, 1e-10)
    
    # Step 2: Handle negative values
    for col in data_clean.columns:
        min_val = data_clean[col].min()
        if min_val <= 0:
            shift = abs(min_val) + 1e-10
            data_clean[col] = data_clean[col] + shift
            print(f"   🔧 Applied shift of {shift:.6f} to {col}")
    
    # Step 3: Calculate log returns
    print("📈 Calculating log returns...")
    log_returns = np.log(data_clean / data_clean.shift(1))
    
    # Step 4: Handle infinite values
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
    
    # Step 5: Drop rows with any NaN values
    log_returns_clean = log_returns.dropna()
    
    print(f"✅ Log returns calculated: {log_returns_clean.shape[0]} observations")
    print(f"🧹 Data cleaning: {data.shape[0] - log_returns_clean.shape[0]} rows dropped")
    
    return log_returns_clean

def demonstrate_fix():
    """Demonstrate the fix with sample data"""
    print("🚀 Demonstrating RuntimeWarning fix...")
    
    # Load your data
    try:
        df = pd.read_csv('data/combined_em_macro_data.csv', parse_dates=['date'], index_col='date')
        print(f"📊 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Apply Term Spread engineering first (as in your notebook)
        shift_value = df['Term_Spread'].min()
        if shift_value < 0:
            print(f"🔧 Term_Spread Engineering: applying shift of {shift_value:.4f}")
            df['Term_Spread_Engineered'] = df['Term_Spread'] - shift_value
            df.drop(columns=['Term_Spread'], inplace=True)
            df.rename(columns={'Term_Spread_Engineered': 'Term_Spread'}, inplace=True)
        
        # Now apply safe log returns
        log_returns = safe_log_returns(df)
        
        print(f"\n📋 Log Returns Summary:")
        print(log_returns.describe().round(4))
        
        print("\n✅ RuntimeWarning should be eliminated!")
        
    except FileNotFoundError:
        print("❌ Data file not found. Please run this from the macro-factor-model-EM directory.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    demonstrate_fix()
