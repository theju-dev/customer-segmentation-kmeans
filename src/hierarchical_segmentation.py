import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage,dendrogram
DATA_PATH="data/Mall_Customers.csv"
FEATURES=["Annual Income (k$)","Spending Score (1-100)"]
FINAL_N_CLUSTERS=5
CLUSTER_NAMES = {
    0: "High income - low spending",
    1: "High income - high spending",
    2: "Medium income - medium spending",
    3: "Low income - high spending",
    4: "Low income - low spending"
}
SEGMENTATION_OUTPUT_PATH="outputs/hierarchical_customer_segments.csv"
PROFILE_OUTPUT_PATH="outputs/hierarchical_cluster_profile.csv"
def load_data(DATA_PATH):
    customers_df=pd.read_csv(DATA_PATH)
    return customers_df
def prepare_features(customers_df):
    X=customers_df[FEATURES].copy()
    return X
def scale_features(X):
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    return X_scaled,scaler
def plot_dendrogram(X_scaled):
    linkage_matrix=linkage(X_scaled,method="ward")
    plt.figure(figsize=(12,6))
    dendrogram(linkage_matrix)
    plt.xlabel("Customers")
    plt.ylabel("Distance")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()
def evaluate_clusters(X_scaled):
    evaluation_results=[]
    for n_clusters in range(2,7):
        hierarchical_model=AgglomerativeClustering(n_clusters=n_clusters,linkage="ward")
        cluster_labels=hierarchical_model.fit_predict(X_scaled)
        silhouette=silhouette_score(X_scaled,cluster_labels)
        evaluation_results.append({"n_clusters":n_clusters,"silhouette_score":silhouette})
    evaluation_df=pd.DataFrame(evaluation_results)
    return evaluation_df
def fit_hierarchical_model(X_scaled,n_clusters):
     hierarchical_model=AgglomerativeClustering(n_clusters=n_clusters,linkage="ward")
     cluster_labels=hierarchical_model.fit_predict(X_scaled)
     return hierarchical_model,cluster_labels
def assign_cluster_labels(customer_df,cluster_labels):
    segmented_customers_df=customer_df.copy()
    segmented_customers_df["Hierarchical_Cluster"]=cluster_labels
    return segmented_customers_df
def create_cluster_profile(segmented_customers_df):
    cluster_profile_df=segmented_customers_df.groupby("Hierarchical_Cluster").agg(customer_count=("CustomerID","count"),average_income=("Annual Income (k$)","mean"),average_spending=("Spending Score (1-100)","mean")).round(2)
    return cluster_profile_df
def assign_cluster_names(segmented_cluster_df):
    segmented_cluster_df["Customer Segment"]=segmented_cluster_df["Hierarchical_Cluster"].map(CLUSTER_NAMES)
    return segmented_cluster_df
def plot_customer_segments(segmented_customers_df):
    plt.figure(figsize=(12,8))
    plt.scatter(segmented_customers_df["Annual Income (k$)"],
                segmented_customers_df["Spending Score (1-100)"],
                label="Hierachical clustering",
                c=segmented_customers_df["Hierarchical_Cluster"],
                cmap="viridis",
                alpha=0.7)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Hierarchical customer segments")
    plt.colorbar()
    plt.legend()
    plt.show()
def save_results(segmented_customers_df,SEGMENTATION_OUTPUT_PATH,cluster_profile_df,PROFILE_OUTPUT_PATH):
    segmented_customers_df.to_csv(SEGMENTATION_OUTPUT_PATH,index=False)
    cluster_profile_df.reset_index().to_csv(PROFILE_OUTPUT_PATH,index=False)
def calculate_final_silhouette_score(X_scaled,cluster_labels):
    silhouette=silhouette_score(X_scaled,cluster_labels)
    return silhouette
def main():
    customers_df=load_data(DATA_PATH)
    X=prepare_features(customers_df)
    X_scaled,scaler=scale_features(X)
    plot_dendrogram(X_scaled)
    evaluation_df=evaluate_clusters(X_scaled)
    print("\nHierarchical clustering evaluation")
    print(evaluation_df)
    hierarchical_model, cluster_labels=fit_hierarchical_model(X_scaled,FINAL_N_CLUSTERS)
    final_silhouette=calculate_final_silhouette_score(X_scaled,cluster_labels)
    segmented_customers_df=assign_cluster_labels(customers_df,cluster_labels)
    print("\n Final hierarchical silhouette score:",round(final_silhouette))
    print("\nSegmented customers")
    print(segmented_customers_df.head())
    cluster_profile_df=create_cluster_profile(segmented_customers_df)
    print("\nHierarchical cluster profile")
    print(cluster_profile_df)
    segmented_customers_df=assign_cluster_names(segmented_customers_df)
    print("\n Customers with business segment names")
    print(segmented_customers_df.head())
    plot_customer_segments(segmented_customers_df)
    save_results(segmented_customers_df,SEGMENTATION_OUTPUT_PATH,cluster_profile_df,PROFILE_OUTPUT_PATH)

if __name__=="__main__":
    main()
