import joblib
import pandas as pd
model_path="models/kmeans_customer_segmentation.joblib"
kmeans_pipeline=joblib.load(model_path)
cluster_names={0: "Medium income - medium spending",
    1: "High income - high spending",
    2: "Low income - high spending",
    3: "High income - low spending",
    4: "Low income - low spending"
}
def predict_customer_segment(annual_income,spending_score):
    customer_data=pd.DataFrame({"Annual Income (k$)":[annual_income],
                                "Spending Score (1-100)":[spending_score]})
    predicted_cluster=kmeans_pipeline.predict(customer_data)
    cluster_id=int(predicted_cluster[0])
    segment_name=cluster_names[cluster_id]
    return {"cluster":cluster_id,
            "segment":segment_name}

if __name__=="__main__":
    result=predict_customer_segment(annual_income=90,spending_score=85)
    print("\nPrediction result")
    print(result)