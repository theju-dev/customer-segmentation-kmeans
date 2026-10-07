# Customer Segmentation Using Machine Learning

## Project Overview

This project applies unsupervised machine learning techniques to segment customers based on their purchasing characteristics.

The objective is to identify meaningful customer groups that can support targeted marketing, customer engagement, and business decision-making.

The project explores and compares three clustering approaches:

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN

## Dataset

The analysis uses customer information including features such as:

- Annual Income
- Spending Score

These features are used to identify customers with similar purchasing and spending characteristics.

## Project Workflow

The project follows the following workflow:

1. Load and inspect the customer dataset.
2. Perform data preprocessing and feature selection.
3. Scale the selected numerical features.
4. Apply K-Means clustering.
5. Determine an appropriate number of clusters using clustering evaluation techniques.
6. Analyze and visualize the K-Means customer segments.
7. Apply Hierarchical Clustering and analyze the resulting cluster structure.
8. Apply DBSCAN to identify density-based customer groups and noise points.
9. Compare the behavior and results of the different clustering approaches.
10. Interpret the identified customer segments from a business perspective.

## K-Means Clustering

K-Means groups customers by assigning each customer to the nearest learned cluster centroid.

The selected customer features are standardized before clustering so that differences in feature scales do not dominate the distance calculations.

The project also demonstrates how a fitted scaler and K-Means model can be saved and reused for assigning new customers to learned clusters.

## Hierarchical Clustering

Hierarchical Clustering is used to study how customers can be progressively grouped based on similarity.

Unlike K-Means, hierarchical clustering builds a hierarchy of clusters rather than immediately assigning observations around predefined centroids.

The hierarchical structure provides another perspective on the natural grouping of customers.

## DBSCAN

DBSCAN is a density-based clustering algorithm.

Instead of requiring a predefined number of clusters, DBSCAN forms clusters based on dense regions of data.

It can also identify observations that do not belong to sufficiently dense regions as noise or outliers.

This makes DBSCAN useful for exploring irregular cluster structures and unusual customer observations.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SciPy
- Joblib

## Business Interpretation

The identified customer groups can support different business strategies.

For example:

- High-income, high-spending customers may be candidates for loyalty or premium programs.
- High-income, low-spending customers may represent an opportunity for increased engagement.
- Low-income, high-spending customers show relatively strong spending behavior despite lower income.
- Medium-income, medium-spending customers represent the central customer group.
- Low-income, low-spending customers may require different engagement strategies.

These interpretations are based primarily on the customer features used for clustering and should not be treated as complete customer profiles.

## Key Learning Outcomes

This project demonstrates:

- Practical implementation of unsupervised machine learning.
- Feature scaling for distance-based algorithms.
- K-Means clustering and cluster interpretation.
- Hierarchical clustering and hierarchical customer grouping.
- DBSCAN and density-based clustering.
- Identification of potential noise or outlier observations.
- Comparison of different clustering approaches.
- Translation of clustering results into business-oriented insights.