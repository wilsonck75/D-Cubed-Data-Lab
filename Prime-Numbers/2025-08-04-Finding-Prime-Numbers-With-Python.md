---
layout: post
title: "Finding Prime Numbers with Python: A Comprehensive Data Science Approach"
date: 2025-08-04
categories: [Python, Data Science, Mathematics]
tags: [Python, Primes, Data Analysis, Visualization, Jupyter]
---

# Finding Prime Numbers with Python: A Comprehensive Data Science Approach

_8 minute read_

In this post, I'm going to walk you through a comprehensive data science approach to finding and analyzing prime numbers using Python. Unlike simple prime-finding algorithms, this implementation includes statistical analysis, advanced visualizations, and professional data export capabilities - making it a complete toolkit for prime number research and education.

If you're not sure what a prime number is, it's a natural number greater than 1 that has exactly two positive divisors: 1 and itself. For example, 7 is prime because only 7 and 1 divide evenly into it, while 8 is not prime because 2 and 4 also divide into it.

Let's dive into building a professional-grade prime number analysis tool!

## What Makes a Number Prime?

A **prime number** is a natural number greater than 1 that has exactly two positive divisors: 1 and itself. Examples include 2, 3, 5, 7, 11, 13, 17, 19, 23, etc.

## Algorithm Overview: Trial Division Method

Our implementation uses the **trial division method** with several key optimizations:

### 1. Square Root Optimization 📐

Instead of checking all numbers up to `n`, we only check up to `√n`. Here's why:

- If `n = a × b` where `a ≤ b`, then `a ≤ √n`
- If we find no divisors up to `√n`, there can't be any larger ones
- **Time Complexity**: Reduces from O(n) to O(√n) - a massive improvement!

**Example**: To check if 49 is prime, we only test divisors up to √49 = 7, not all the way to 49.

### 2. Why Skip Even Numbers? ⚡

While our current algorithm doesn't implement this optimization, advanced versions skip even numbers because:

- **Only 2 is an even prime** - all other even numbers are divisible by 2
- After checking 2, we can test only odd candidates: 3, 5, 7, 9, 11, etc.
- **Speed Improvement**: Cuts the search space in half!

### 3. Algorithm Efficiency Comparison

| Method                    | Time Complexity | Space   | Best For                         |
| ------------------------- | --------------- | ------- | -------------------------------- |
| **Trial Division**        | O(n√n)          | O(π(n)) | Small to medium ranges           |
| **Sieve of Eratosthenes** | O(n log log n)  | O(n)    | Large ranges, all primes up to n |
| **Segmented Sieve**       | O(n log log n)  | O(√n)   | Very large ranges                |

## Setting Up the Environment

First, let's import all the necessary libraries for our comprehensive analysis:

```python
# Standard library imports
import sys
import logging
from pathlib import Path

# Mathematical and data processing libraries
import math
import statistics
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
```

## Core Prime Detection Algorithm

Here's our optimized prime detection function:

```python
def is_prime(num):
    """
    Check if a given number is prime using trial division method.

    Algorithm Explanation:
        - Numbers ≤ 1 are not prime by definition
        - We only need to check divisors up to √num because:
          * If num = a × b and a ≤ b, then a ≤ √num
          * This reduces the time complexity from O(n) to O(√n)
    """
    if num <= 1:
        return False

    # Check for divisibility from 2 to √num
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:  # If num is divisible by i, it's not prime
            return False

    return True

def generate_primes(limit):
    """
    Generate all prime numbers up to and including the given limit.

    Time Complexity: O(n√n) where n is the limit
    Space Complexity: O(π(n)) where π(n) is the prime counting function
    """
    print(f"Starting prime number generation for limit = {limit}")

    primes = []  # List to store found prime numbers

    # Iterate through all numbers from 2 to limit (inclusive)
    for candidate in range(2, limit + 1):
        if is_prime(candidate):
            primes.append(candidate)

        # Log progress for large numbers (every 1000 numbers checked)
        if candidate % 1000 == 0:
            print(f"Checked up to {candidate}, found {len(primes)} primes so far")

    print(f"Prime generation complete! Found {len(primes)} prime numbers up to {limit}")

    return primes
```

## Configurable Analysis

One of the key features of this implementation is easy configuration:

```python
# SET YOUR UPPER LIMIT HERE:
n = 1000  # <- CHANGE THIS NUMBER TO YOUR DESIRED UPPER LIMIT

# Generate all prime numbers up to the specified limit
print(f"Generating prime numbers up to {n}...")
primes = generate_primes(n)

print(f"Generation Complete!")
print(f"Found {len(primes)} prime numbers between 2 and {n}")
print(f"Smallest prime: {primes[0] if primes else 'None'}")
print(f"Largest prime: {primes[-1] if primes else 'None'}")
```

For n = 1000, this generates 168 prime numbers, with the largest being 997.

## Statistical Analysis

Our implementation goes beyond simple prime generation to provide comprehensive statistical analysis:

```python
def analyze_primes(prime_list):
    """
    Perform statistical analysis on the generated prime numbers.
    """
    if not prime_list:
        return {"error": "No prime numbers to analyze"}

    stats = {
        "count": len(prime_list),
        "min": min(prime_list),
        "max": max(prime_list),
        "mean": statistics.mean(prime_list),
        "median": statistics.median(prime_list),
        "range": max(prime_list) - min(prime_list)
    }

    # Calculate gaps between consecutive primes
    gaps = [prime_list[i+1] - prime_list[i] for i in range(len(prime_list)-1)]
    if gaps:
        stats["avg_gap"] = statistics.mean(gaps)
        stats["max_gap"] = max(gaps)
        stats["min_gap"] = min(gaps)

    return stats

# Create a comprehensive DataFrame with additional information
df_primes = pd.DataFrame({
    'Index': range(len(primes)),
    'Prime_Number': primes,
    'Is_Even': [p == 2 for p in primes],
    'Digit_Count': [len(str(p)) for p in primes],
    'Last_Digit': [p % 10 for p in primes]
})

# Add gap analysis (difference from previous prime)
gaps = [0] + [primes[i] - primes[i-1] for i in range(1, len(primes))]
df_primes['Gap_From_Previous'] = gaps
```

This analysis reveals interesting patterns:

- **Total Prime Numbers Found**: 168 for numbers up to 1,000
- **Average Gap Between Primes**: ~5.95
- **Largest Gap**: 20 (between 887 and 907)
- **Distribution**: 4 single-digit primes (23.8% of all primes up to 1,000)

## Advanced Visualizations

The implementation includes comprehensive visualizations with 6 different charts:

```python
def create_prime_visualizations(df_primes, limit):
    """
    Create comprehensive visualizations for prime number analysis.
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Prime Numbers Analysis (up to {limit:,})', fontsize=20, fontweight='bold', y=0.98)

    # 1. Main scatter plot of prime numbers with trend line
    ax1 = plt.subplot(2, 3, (1, 2))  # Spans 2 columns
    sns.scatterplot(data=df_primes, x='Index', y='Prime_Number',
                   s=50, alpha=0.7, color='darkblue', edgecolor='white', linewidth=0.5)

    # Add trend line
    z = np.polyfit(df_primes['Index'], df_primes['Prime_Number'], 1)
    p = np.poly1d(z)
    ax1.plot(df_primes['Index'], p(df_primes['Index']), "r--", alpha=0.8, linewidth=2, label='Trend Line')

    # 2. Gap analysis histogram
    ax2 = plt.subplot(2, 3, 3)
    gaps = df_primes['Gap_From_Previous'][1:]  # Exclude first gap (which is 0)
    ax2.hist(gaps, bins=min(30, len(set(gaps))), alpha=0.7, color='green', edgecolor='black')

    # 3. Last digit distribution
    ax3 = plt.subplot(2, 3, 4)
    last_digit_counts = df_primes['Last_Digit'].value_counts().sort_index()
    bars = ax3.bar(last_digit_counts.index, last_digit_counts.values,
                   color=['red' if x == 2 else 'blue' if x == 5 else 'green' for x in last_digit_counts.index],
                   alpha=0.7, edgecolor='black')

    # 4. Prime density over ranges
    ax4 = plt.subplot(2, 3, 5)
    # Calculate prime density in ranges
    range_size = max(100, limit // 10)
    ranges = list(range(0, limit + range_size, range_size))
    densities = []

    for i in range(len(ranges) - 1):
        start, end = ranges[i], ranges[i + 1]
        primes_in_range = df_primes[(df_primes['Prime_Number'] >= start) &
                                  (df_primes['Prime_Number'] < end)]['Prime_Number'].count()
        density = primes_in_range / range_size * 100  # Percentage
        densities.append(density)

    ax4.plot(range(len(densities)), densities, marker='o', linewidth=2, markersize=6, color='purple')

    # 5. Cumulative count
    ax5 = plt.subplot(2, 3, 6)
    ax5.plot(df_primes['Prime_Number'], df_primes['Index'] + 1, linewidth=2, color='orange')

    plt.tight_layout()
    plt.show()

    return fig
```

These visualizations reveal fascinating patterns:

- **Distribution**: Shows how primes become sparser as numbers get larger
- **Gap Analysis**: Most gaps are small (2-6), but some reach 20
- **Last Digit Patterns**: Most primes end in 1, 3, 7, or 9 (except 2 and 5)
- **Density Trends**: Prime density decreases as we move to higher ranges

## Professional Data Export

The implementation includes comprehensive data export capabilities:

```python
def export_data_and_visualizations(df_primes, fig, limit):
    """
    Export the generated data and visualizations to organized folders.
    """
    export_results = {}

    try:
        # Create output directories
        data_dir = Path("../data")
        plots_dir = Path("../plots")
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export DataFrame to CSV
        csv_path = data_dir / f"prime_numbers_up_to_{limit}.csv"
        df_primes.to_csv(csv_path, index=False)

        # 2. Export to Excel with multiple sheets
        excel_path = data_dir / f"prime_numbers_analysis_{limit}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_primes.to_excel(writer, sheet_name='Prime_Numbers', index=False)

            # Summary statistics sheet
            summary_stats = pd.DataFrame([
                ['Total Primes Found', len(df_primes)],
                ['Upper Limit', limit],
                ['Smallest Prime', df_primes['Prime_Number'].min()],
                ['Largest Prime', df_primes['Prime_Number'].max()],
                ['Average Prime Value', df_primes['Prime_Number'].mean()],
                ['Median Prime Value', df_primes['Prime_Number'].median()],
                ['Average Gap', df_primes['Gap_From_Previous'][1:].mean()],
                ['Maximum Gap', df_primes['Gap_From_Previous'].max()],
            ], columns=['Statistic', 'Value'])
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)

        # 3. Export visualization to high-quality PNG
        png_path = plots_dir / f"prime_numbers_visualization_{limit}.png"
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

        # 4. Export visualization to PDF
        pdf_path = plots_dir / f"prime_numbers_visualization_{limit}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')

        # 5. Create a summary report text file
        report_path = data_dir / f"prime_numbers_report_{limit}.txt"
        with open(report_path, 'w') as f:
            f.write(f"PRIME NUMBERS ANALYSIS REPORT\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"PARAMETERS:\n")
            f.write(f"Upper Limit: {limit:,}\n\n")
            # ... additional report content

    except Exception as e:
        print(f"Export error: {str(e)}")

    return export_results
```

## Performance Insights

Our implementation prioritizes **clarity and educational value** over maximum speed. Here's what you can expect:

- **n = 100**: Very fast (~25 primes)
- **n = 500**: Fast (~95 primes)
- **n = 1,000**: Quick (~168 primes)
- **n = 5,000**: Moderate (~669 primes)
- **n = 10,000**: Slower (~1,229 primes)

For production use with very large numbers, consider:

- Sieve of Eratosthenes for finding all primes up to a limit
- Miller-Rabin test for checking individual large numbers
- Specialized libraries like `sympy` for advanced prime operations

## Fun Prime Facts! 🎯

- **Euclid's Theorem**: There are infinitely many prime numbers
- **Prime Number Theorem**: Approximately n/ln(n) numbers less than n are prime
- **Goldbach's Conjecture**: Every even number > 2 can be expressed as the sum of two primes
- **Twin Primes**: Pairs like (3,5), (5,7), (11,13) that differ by 2

## Key Results for n = 1,000

Running our comprehensive analysis for all primes up to 1,000 reveals:

- **168 prime numbers** between 2 and 1,000
- **Largest prime**: 997
- **Average gap**: 5.95
- **Largest gap**: 20 (between 887 and 907)
- **Single-digit primes**: 4 (representing 2.4% of all primes up to 1,000)

## Conclusion

This implementation demonstrates how a simple mathematical concept like prime numbers can be transformed into a comprehensive data science project. By combining efficient algorithms, statistical analysis, advanced visualizations, and professional data export capabilities, we've created a tool that's both educational and practical.

The modular design makes it easy to:

- Adjust the upper limit for different analyses
- Extend the statistical analysis
- Customize the visualizations
- Export data in multiple formats for further research

Whether you're a student learning about prime numbers, a researcher studying number theory, or a data scientist exploring mathematical patterns, this comprehensive approach provides a solid foundation for prime number analysis.

I hope you enjoyed this deep dive into finding and analyzing prime numbers with Python!

---

## Downloads

**Access the complete project files:**

- **📓 Jupyter Notebook**: [Prime-Numbers.ipynb](./notebooks/Prime-Numbers.ipynb) - Complete interactive notebook with all code and analysis
- **📊 Data Files**:
  - [CSV Data](./data/prime_numbers_up_to_1000.csv) - Raw prime numbers data
  - [Excel Analysis](./data/prime_numbers_analysis_1000.xlsx) - Multi-sheet workbook with statistics
  - [Text Report](./data/prime_numbers_report_1000.txt) - Summary analysis report
- **🎨 Visualizations**:
  - [High-Resolution PNG](./plots/prime_numbers_visualization_1000.png) - 300 DPI visualization charts
  - [PDF Version](./plots/prime_numbers_visualization_1000.pdf) - Vector graphics for publications

**GitHub Repository**: [D-Cubed-Data-Lab/Prime-Numbers](https://github.com/wilsonck75/D-Cubed-Data-Lab/tree/main/Prime-Numbers)

---

_This post is part of the D³ Data Lab series exploring data science applications in mathematics and beyond. Follow for more data-driven insights!_
