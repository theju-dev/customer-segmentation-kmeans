import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
DATA_PATH="data/Mall_Customers.csv"
FEATURES=["Annual Income (k$)","Spending Score (1-100)"]
FINAL_EPS=0.35
FINAL_MIN_SAMPLES=4
OUTPUT_PATH = "outputs/dbscan_customer_segments.csv"
PROFILE_OUTPUT_PATH = "outputs/dbscan_cluster_profile.csv"
def load_data(DATA_PATH):
    customers_df=pd.read_csv(DATA_PATH)
    return customers_df
def prepare_features(customers_df):
    X=customers_df[FEATURES]
    return X
def scale_features(X):
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    return X_scaled,scaler
def plot_k_distance(X_scaled,min_samples):
    neighbors_model=NearestNeighbors(n_neighbors=min_samples)
    neighbors_model.fit(X_scaled)
    distances,indices=neighbors_model.kneighbors(X_scaled)
    k_distances=distances[:,-1]
    k_distances=np.sort(k_distances)
    plt.figure(figsize=(12,8))
    plt.plot(k_distances)
    plt.xlabel("Customers sorted by distance")
    plt.ylabel("Distances to kth nearest neighbor")
    plt.title("K-distance plot for DBSCAN")
    plt.show()
def evaluate_dbscan(X_scaled,eps_values,min_samples):
     evaluation_results=[]
     for eps in eps_values:
          dbscan_model=DBSCAN(eps=eps,min_samples=min_samples)
          cluster_labels=dbscan_model.fit_predict(X_scaled)
          unique_labels=set(cluster_labels)
          n_clusters=len(unique_labels-{-1})
          n_noise=list(cluster_labels).count(-1)
          non_noise_mask=cluster_labels!=-1
          if n_clusters>=2:
               silhouette=silhouette_score(X_scaled[non_noise_mask],cluster_labels[non_noise_mask])
          else:
               silhouette=None
          evaluation_results.append({"eps":eps,"n_clusters":n_clusters,"n_noise":n_noise,"silhouette_score":silhouette})
     evaluation_df=pd.DataFrame(evaluation_results)
     return evaluation_df
def get_cluster_sizes(cluster_labels):
     cluster_sizes=pd.Series(cluster_labels).value_counts().sort_index()
     return cluster_sizes
def fit_dbscan_model(X_scaled,eps,min_samples):
     dbscan_model=DBSCAN(eps=eps,min_samples=min_samples)
     cluster_labels=dbscan_model.fit_predict(X_scaled)
     return dbscan_model,cluster_labels
def assign_cluster_labels(customers_df,cluster_labels):
     segmented_customers_df=customers_df.copy()
     segmented_customers_df["DBSCAN_Cluster"]=cluster_labels
     return segmented_customers_df
def create_cluster_profile(segmented_customers_df):
     cluster_profile_df=segmented_customers_df.groupby("DBSCAN_Cluster").agg(customer_count=("CustomerID","count"),average_income=("Annual Income (k$)","mean"),average_spending=("Spending Score (1-100)","mean")).round(2)
     return cluster_profile_df
def plot_dbscan_clusters(segmented_customers_df,eps):
     plt.figure(figsize=(10,6))
     plt.scatter(segmented_customers_df["Annual Income (k$)"],
                 segmented_customers_df["Spending Score (1-100)"],
                 c=segmented_customers_df["DBSCAN_Cluster"],
                 cmap="viridis",
                 alpha=0.7)
     plt.xlabel("Annual Income (k$)")
     plt.ylabel("Spending Score (1-100)")
     plt.title(f"DBSCAN Customer Segments -eps={eps}")
     plt.colorbar(label="Cluster")
     plt.show()
def save_results(segmented_customers_df,cluster_profile_df,segmentation_output_path,
    profile_output_path):
     segmented_customers_df.to_csv(segmentation_output_path,index=False)
     cluster_profile_df.reset_index().to_csv(profile_output_path,index=False)
def main():
    customer_df=load_data(DATA_PATH)
    X=prepare_features(customer_df)
    X_scaled,scaler=scale_features(X)
    plot_k_distance(X_scaled,min_samples=4)
    eps_values=[0.35,0.40,0.45,0.50,0.55]
    evaluation_df=evaluate_dbscan(X_scaled,eps_values,min_samples=4)
    print("\nDBSCAN EVALAUTION")
    print(evaluation_df)

   # inspection_model=DBSCAN(eps=0.35,min_samples=4)
   # inspection_labels=inspection_model.fit_predict(X_scaled)
   # cluster_sizes=get_cluster_sizes(inspection_labels)
   # print("\n DBSCAN CLuster sizes for eps=0.35")
   # print(cluster_sizes)
    for eps in [0.35,0.40]:
         inspection_model=DBSCAN(eps=eps,min_samples=4)
         inspection_labels=inspection_model.fit_predict(X_scaled)
         cluster_sizes=get_cluster_sizes(inspection_labels)
         print(f"\n DBSCAN cluster sizes for eps={eps}:")
         print(cluster_sizes)
    for eps in [0.35,0.40]:
         candidate_model,candidate_labels=fit_dbscan_model(X_scaled,eps,min_samples=4)
         candidate_customers_df=assign_cluster_labels(customer_df,candidate_labels)
         candidate_profile_df=create_cluster_profile(candidate_customers_df)
         print(f"\n DBSCAN Cluster profile for eps ={eps}")
         print(candidate_profile_df)
         plot_dbscan_clusters(candidate_customers_df,eps)
    dbscan_model,cluster_labels=fit_dbscan_model(X_scaled,FINAL_EPS,FINAL_MIN_SAMPLES)
    segmented_customers_df=assign_cluster_labels(customer_df,cluster_labels)
    cluster_profile_df=create_cluster_profile(segmented_customers_df)
    print("\n Final DBSCAN Cluster Profile")
    print(cluster_profile_df)
    save_results(segmented_customers_df,cluster_profile_df,OUTPUT_PATH,PROFILE_OUTPUT_PATH)
if __name__=="__main__":
        main()