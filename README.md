# Customer Segmentation Using K-Means Clustering

## Project Overview

This project performs customer segmentation using K-Means clustering.

The goal is to group customers based on:

- Annual Income
- Spending Score

These customer segments can help businesses understand different purchasing and spending patterns and support targeted marketing strategies.

## Dataset

The project uses the Mall Customers dataset.

The dataset contains the following columns:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

For clustering, the following features are used:

- Annual Income (k$)
- Spending Score (1-100)

CustomerID is excluded because it is only an identifier and does not represent meaningful customer similarity.

## Project Workflow

The project follows these steps:

1. Load customer data
2. Inspect the dataset
3. Check missing values and duplicate records
4. Perform exploratory visualization
5. Select clustering features
6. Standardize features using StandardScaler
7. Evaluate different values of K
8. Compare inertia using the Elbow Method
9. Compare Silhouette Scores
10. Select the final number of clusters
11. Train the final K-Means model
12. Assign customers to clusters
13. Profile and interpret the clusters
14. Add business-readable segment names
15. Visualize the final customer segments
16. Save the segmented dataset
17. Save the trained preprocessing and clustering pipeline
18. Use the saved pipeline to assign new customers to existing segments

## Model Selection

Different values of K from 2 to 10 are evaluated.

Two metrics are used:

### Inertia

Inertia measures how close customers are to the centroid of their assigned cluster.

The Elbow Method is used to identify where increasing the number of clusters provides diminishing improvement.

### Silhouette Score

The Silhouette Score evaluates both:

- Cohesion within a cluster
- Separation between clusters

Based on the Elbow Method, Silhouette Score, and interpretability of the resulting groups, the final number of clusters is:

K = 5

## Customer Segments

The final model produces five customer groups:

| Cluster | Customer Segment |
|---|---|
| 0 | Medium income - medium spending |
| 1 | High income - high spending |
| 2 | Low income - high spending |
| 3 | High income - low spending |
| 4 | Low income - low spending |

Cluster numbers are model-generated identifiers. Business-readable names are assigned after examining the characteristics of each cluster.

## Machine Learning Pipeline

The final model uses a Scikit-learn Pipeline containing:

1. StandardScaler
2. KMeans

The scaler ensures that both clustering features contribute appropriately to distance calculations.

The complete fitted pipeline is saved so the same preprocessing and clustering logic can be reused for new customers.

## Project Structure

Customer_Segmentation/
│
├── data/
│   └── Mall_Customers.csv
│
├── models/
│   └── kmeans_customer_segmentation.joblib
│
├── outputs/
│   └── customer_segments.csv
│
├── src/
│   ├── customer_segmentation.py
│   └── predict_segment.py
│
├── .gitignore
├── README.md
└── requirements.txt

## Training

Run the customer segmentation workflow:

python src/customer_segmentation.py

The training workflow:

- loads and validates the dataset
- evaluates different K values
- trains the final K-Means pipeline
- assigns customer clusters
- creates business-readable segment names
- saves the segmented customer dataset
- saves the fitted model pipeline

## New Customer Segmentation

Run:

python src/predict_segment.py

Example customer:

Annual Income = 90 k$
Spending Score = 85

Example result:

{
    "cluster": 1,
    "segment": "High income - high spending"
}

The saved StandardScaler first transforms the new customer's features.

The fitted K-Means model then assigns the customer to the nearest learned cluster centroid.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

## Business Interpretation

The identified customer groups can support different business strategies.

For example:

- High-income, high-spending customers may be candidates for loyalty or premium programs.
- High-income, low-spending customers may represent an opportunity for increased engagement.
- Low-income, high-spending customers show relatively strong spending behavior despite lower income.
- Medium-income, medium-spending customers represent the central customer group.
- Low-income, low-spending customers may require different engagement strategies.

These interpretations are based only on Annual Income and Spending Score and should not be treated as complete customer profiles.