# colab

import numpy as np
import pandas as pd 
train_df = pd.read_csv("../data/train.csv")
train_df

#itemizing
input_features = list(train_df.columns)
input_features.remove("Ticket")
input_features.remove("PassengerId")
input_features.remove("Survived")

print(f"Input features: {input_features}")


#exchange for tensorflow
import tensorflow as tf
import tensorflow_decision_forests as tfdf

def tokenize_names(features, labels=None):
    """Divite the names into tokens. TF-DF can consume text tokens natively."""
    features["Name"] =  tf.strings.split(features["Name"])
    return features, labels

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df,label="Survived").map(tokenize_names)


#training
model = tfdf.keras.GradientBoostedTreesModel(
    verbose=0, # Very few logs
    features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
    exclude_non_specified_features=True, # Only use the features in "features"
    random_seed=1234,
)
model.fit(train_ds)

self_evaluation = model.make_inspector().evaluation()
print(f"Accuracy: {self_evaluation.accuracy} Loss:{self_evaluation.loss}")

#Prediction
serving_df = pd.read_csv("../data/test.csv")
serving_ds = tfdf.keras.pd_dataframe_to_tf_dataset(serving_df).map(tokenize_names)

def prediction_to_kaggle_format(model, threshold=0.5):
    proba_survive = model.predict(serving_ds, verbose=0)[:,0]
    return pd.DataFrame({
        "PassengerId": serving_df["PassengerId"],
        "Survived": (proba_survive >= threshold).astype(int)
    })
"""
def make_submission(kaggle_predictions):
    path="/kaggle/working/submission.csv"
    kaggle_predictions.to_csv(path, index=False)
    print(f"Submission exported to {path}")
    
kaggle_predictions = prediction_to_kaggle_format(model)
make_submission(kaggle_predictions)

"""