import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
DATA_PATH="data/Mall_Customers.csv"
FEATURES=["Annual Income (k$)",
    "Spending Score (1-100)"
]
CLUSTER_NAMES = {
    0: "Medium income - medium spending",
    1: "High income - high spending",
    2: "Low income - high spending",
    3: "High income - low spending",
    4: "Low income - low spending"
}
OUTPUT_PATH="outputs/customer_segments.csv"
MODEL_PATH="models/kmeans_customer_segmentation.joblib"
##new changes
def load_data(DATA_PATH):
    customers_df=pd.read_csv(DATA_PATH)
    return customers_df
#old one 
#customers_df=pd.read_csv(DATA_PATH)
def inspect_data(customers_df):
    print(customers_df.head())
    print(customers_df.shape)
    customers_df.info()
    print("\nMissing values")
    print(customers_df.isnull().sum())
    print("\n Duplicate rows")
    print(customers_df.duplicated().sum())
    print("\nNumerical summary")
    print(customers_df.describe())

#old one 
'''print(customers_df.head())
print(customers_df.shape)
customers_df.info()
print("\nMissing values")
print(customers_df.isnull().sum())
print("\n Duplicate rows")
print(customers_df.duplicated().sum())
print("\nNumerical summary")
print(customers_df.describe())'''

#new one
def plot_customer_distribution(customers_df):
    plt.figure(figsize=(8,6))
    plt.scatter(customers_df["Annual Income (k$)"],customers_df["Spending Score (1-100)"])
    plt.xlabel("Annual income (k$)")
    plt.ylabel("Spending score (1-100)")
    plt.title("Customer distribution:Income and spending score")
    plt.show()

#old one
'''plt.figure(figsize=(8,6))
plt.scatter(customers_df["Annual Income (k$)"],customers_df["Spending Score (1-100)"])
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score (1-100)")
plt.title("Customer distribution:Income and spending score")
plt.show()
features=["Annual Income (k$)" ,"Spending Score (1-100)"]'''

#new one
def prepare_features(customers_df):
    X=customers_df[FEATURES].copy()
    print("\n Clustering features")
    print(X.head())
    return X

#old one
'''X=customers_df[features].copy()
print("\n Clustering features")
print(X.head())'''

#new one
def evaluate_clusters(X):
    k_values=range(2,11)
    inertia_scores=[]
    silhouette_scores=[]
    for k in k_values:
      kmeans_pipeline=Pipeline(steps=[("scaler",StandardScaler()),("kmeans",KMeans(n_clusters=k,init="k-means++",n_init=10,random_state=42))])
      kmeans_pipeline.fit(X)
      inertia_scores.append(kmeans_pipeline.named_steps["kmeans"].inertia_)
#print(inertia_scores)
      scaler=kmeans_pipeline.named_steps["scaler"]
      X_scaled=scaler.transform(X)
      cluster_labels=kmeans_pipeline.named_steps["kmeans"].labels_
      score=silhouette_score(X_scaled,cluster_labels)
      silhouette_scores.append(score)
    return k_values,inertia_scores,silhouette_scores  

#old one
'''k_values=range(2,11)
inertia_scores=[]
silhouette_scores=[]
#KMeans_pipeline=Pipeline(steps=(["scaler",StandardScaler()],["kmeans",KMeans()]))
for k in k_values:
    kmeans_pipeline=Pipeline(steps=[("scaler",StandardScaler()),("kmeans",KMeans(n_clusters=k,init="k-means++",n_init=10,random_state=42))])
    kmeans_pipeline.fit(X)
    inertia_scores.append(kmeans_pipeline.named_steps["kmeans"].inertia_)
#print(inertia_scores)
    scaler=kmeans_pipeline.named_steps["scaler"]
    X_scaled=scaler.transform(X)
    cluster_labels=kmeans_pipeline.named_steps["kmeans"].labels_
    score=silhouette_score(X_scaled,cluster_labels)
    silhouette_scores.append(score)
print("\nInertia scores")
print(inertia_scores)
print("\n silhouetter scores")
print(silhouette_scores)'''

def plot_cluster_evaluation(k_values,inertia_scores,silhouette_scores):
    plt.figure(figsize=(8,6))
    plt.plot(k_values,inertia_scores,marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow method for optimal K ")
    plt.show()
    plt.figure(figsize=(8,6))
    plt.plot(k_values,silhouette_scores,marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette scores")
    plt.title("Silhouette scores for different K values")
    plt.show()


#old one 
'''plt.figure(figsize=(8,6))
plt.plot(k_values,inertia_scores,marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow method for optimal K ")
plt.show()
plt.figure(figsize=(8,6))
plt.plot(k_values,silhouette_scores,marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette scores")
plt.title("Silhouette scores for different K values")
plt.show()'''

#new one
OPTIMAL_K=5
def train_final_model(X,optimal_k):
    final_kmeans_pipeline=Pipeline(steps=[("scaler",StandardScaler()),
                                      ("kmeans",KMeans(n_clusters=optimal_k,init="k-means++",n_init=10,random_state=42))])
    final_kmeans_pipeline.fit(X)
    return final_kmeans_pipeline

#new one
def assign_clusters(customers_df,final_kmeans_pipeline):
   cluster_labels=final_kmeans_pipeline.named_steps["kmeans"].labels_
   customers_df["Cluster"]=cluster_labels
   return customers_df

#old one 
   ''' cluster_labels=final_kmeans_pipeline.named_steps["kmeans"].labels_


final_kmeans_pipeline=Pipeline(steps=[("scaler",StandardScaler()),
                                      ("kmeans",KMeans(n_clusters=optimal_k,init="k-means++",n_init=10,random_state=42))])
final_kmeans_pipeline.fit(X)
cluster_labels=final_kmeans_pipeline.named_steps["kmeans"].labels_
print(cluster_labels)
customers_df["Cluster"]=cluster_labels
print("\nCustomers with cluster assignments")
print(customers_df)'''

#new one
def profile_clusters(customers_df):
    cluster_sizes=customers_df["Cluster"].value_counts().sort_index()
    cluster_profile=customers_df.groupby("Cluster")[FEATURES].mean()
    print("\n Cluster sizes")
    print(cluster_sizes)
    print("\nCluster profile")
    print(cluster_profile)
    return cluster_sizes, cluster_profile
#old one
'''cluster_sizes=customers_df["Cluster"].value_counts().sort_index()
print("\n Cluster sizes")
print(cluster_sizes)
cluster_profile=customers_df.groupby("Cluster")[features].mean()
print("\n Cluster profile")
print(cluster_profile)
cluster_names={0:"Medium income - medium spending",
               1:"High income - high spending",
               2:"Low income - high spending",
               3:"High income - low spending",
               4:"Low income - low spending"}'''
#NEW ONE
def add_segment_names(customers_df):
    customers_df["Segment"]=customers_df["Cluster"].map(CLUSTER_NAMES)
    return customers_df

#OLD ONE
#customers_df["Segment"]=customers_df["Cluster"].map(cluster_names)
#print("\n Customer with segment names")
#print(customers_df.head())

#new one
def plot_customer_segments(customers_df,OPTIMAL_K):
    plt.figure(figsize=(10,7))
    for cluster_id in range(OPTIMAL_K):
      cluster_data=customers_df[customers_df["Cluster"]==cluster_id]
      plt.scatter(cluster_data["Annual Income (k$)"],cluster_data["Spending Score (1-100)"],label=CLUSTER_NAMES[cluster_id])
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer segments based on income and spending")
    plt.legend()
    plt.show()


#old one
'''plt.figure(figsize=(10,7))
for cluster_id in range(optimal_k):
    cluster_data=customers_df[customers_df["Cluster"]==cluster_id]
    plt.scatter(cluster_data["Annual Income (k$)"],cluster_data["Spending Score (1-100)"],label=cluster_names[cluster_id])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer segments based on income and spending")
plt.legend()
plt.show()'''

#old one
#output_path="outputs/customer_segments.csv"
def save_artifacts(customers_df,final_kmeans_pipeline):
    customers_df.to_csv(OUTPUT_PATH,index=False)
#print(f"\nSegmented customer data saved to : {output_path}")
#model_path="models/kmeans_customer_segmentation.joblib"
    joblib.dump(final_kmeans_pipeline,MODEL_PATH)
    print(f"\nSegmented customer data saved to : {OUTPUT_PATH}")
    print(f"\n Trained k-means pipeline saved to : {MODEL_PATH}")

#old one
'''customers_df.to_csv(output_path,index=False)
print(f"\nSegmented customer data saved to : {output_path}")
#model_path="models/kmeans_customer_segmentation.joblib"
joblib.dump(final_kmeans_pipeline,model_path)
print(f"\n Trained k-means pipeline saved to : {model_path}")'''

def main():
    customers_df = load_data(DATA_PATH)

    inspect_data(customers_df)

    plot_customer_distribution(customers_df)

    X = prepare_features(customers_df)

    k_values, inertia_scores, silhouette_scores = evaluate_clusters(X)

    plot_cluster_evaluation(
        k_values,
        inertia_scores,
        silhouette_scores
    )

    final_kmeans_pipeline = train_final_model(
        X,
        OPTIMAL_K
    )

    customers_df = assign_clusters(
        customers_df,
        final_kmeans_pipeline
    )

    profile_clusters(customers_df)

    customers_df = add_segment_names(customers_df)

    print("\nCustomers with segment names")
    print(customers_df.head())

    plot_customer_segments(
        customers_df,
        OPTIMAL_K
    )

    save_artifacts(
        customers_df,
        final_kmeans_pipeline
    )


if __name__ == "__main__":
    main()