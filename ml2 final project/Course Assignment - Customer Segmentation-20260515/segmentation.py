import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def run_pca(scaled_data, variance_threshold=0.90):
    """Reduces data dimensionality while keeping 90% of the variance information."""
    pca = PCA(n_components=variance_threshold, random_state=42)
    pca_data = pca.fit_transform(scaled_data)
    return pca_data, pca

def get_clustering_metrics(data, k_max=8):
    """Calculates WCSS (Within-Cluster Sum of Squares) to help identify the Elbow."""
    wcss = []
    scores = []
    k_values = range(2, k_max + 1)
    for k in k_values:
        model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = model.fit_predict(data)
        wcss.append(model.inertia_)
        scores.append(silhouette_score(data, labels, sample_size=min(5000, len(data))))
    return k_values, wcss, scores

def apply_final_clustering(pca_data, original_df, k):
    """Applies KMeans with the chosen K value and appends cluster labels to the data."""
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    clusters = kmeans.fit_predict(pca_data)
    result_df = original_df.copy()
    result_df['cluster'] = clusters
    return result_df

def load_basket_data(filepath):
    """Loads and parses the customer_basket CSV, converting list_of_goods from string to list."""
    basket_df = pd.read_csv(filepath)
    basket_df['list_of_goods'] = basket_df['list_of_goods'].apply(eval)
    return basket_df

def get_cluster_rules(basket_df, customer_clusters, cluster_id, min_support=0.05, min_confidence=0.3):
    """
    Mines association rules for customers belonging to a specific cluster.
    Returns top rules sorted by lift.
    """
    # Filter customers in this cluster
    cluster_customers = customer_clusters[customer_clusters['cluster'] == cluster_id]['customer_id']
    cluster_baskets = basket_df[basket_df['customer_id'].isin(cluster_customers)]['list_of_goods'].tolist()

    # Encode transactions
    te = TransactionEncoder()
    te_array = te.fit_transform(cluster_baskets)
    basket_encoded = pd.DataFrame(te_array, columns=te.columns_)

    # Mine frequent itemsets and rules
    frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
    return rules

def get_top_products_per_cluster(basket_df, customer_clusters, cluster_id, top_n=10):
    """Returns the most frequently purchased products for a given cluster."""
    cluster_customers = customer_clusters[customer_clusters['cluster'] == cluster_id]['customer_id']
    cluster_baskets = basket_df[basket_df['customer_id'].isin(cluster_customers)]['list_of_goods']
    all_products = cluster_baskets.explode()
    return all_products.value_counts().head(top_n)