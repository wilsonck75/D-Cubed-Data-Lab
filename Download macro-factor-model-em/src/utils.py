import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_factor_model(X_macro, Y_em, n_components=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_macro)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    betas = {}
    r2 = {}
    for col in Y_em.columns:
        model = LinearRegression().fit(X_pca, Y_em[col])
        betas[col] = model.coef_
        r2[col] = model.score(X_pca, Y_em[col])

    beta_df = pd.DataFrame(betas, index=[f'PC{i+1}' for i in range(n_components)]).T
    r2_df = pd.DataFrame.from_dict(r2, orient='index', columns=['R²'])
    return beta_df, r2_df, pca
