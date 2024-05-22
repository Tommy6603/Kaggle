# colab

import numpy as np
import pandas as pd 
train_df = pd.read_csv("./data/train.csv")
train_df

#itemizing
input_features = list(train_df.columns)
input_features.remove("Ticket")
input_features.remove("PassengerId")
input_features.remove("Survived")

print(f"Input features: {input_features}")
# ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']


#exchange for tensorflow
import tensorflow as tf
# import tensorflow_decision_forests as tfdf

def tokenize_names(features, labels=None): # devide family and given names
    features["Name"] =  tf.strings.split(features["Name"])
    return features, labels

# transfer dataframe from pandas to keras(Target : Survived, Name devided)
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df,label="Survived").map(tokenize_names)


#training
model = tfdf.keras.GradientBoostedTreesModel( # generate instance
    verbose=0, # Very few logs, how detailed is log
    features=[tfdf.keras.FeatureUsage(name=n) for n in input_features], # List of Feature value
    exclude_non_specified_features=True, # Decide feature value you use. 
    random_seed=1234, #control how randome it is. 
)
model.fit(train_ds) # Training
"""
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


"""
def make_submission(kaggle_predictions):
    path="/kaggle/working/submission.csv"
    kaggle_predictions.to_csv(path, index=False)
    print(f"Submission exported to {path}")
    
kaggle_predictions = prediction_to_kaggle_format(model)
make_submission(kaggle_predictions)

"""


"""
Internal process of .fit
1.Obtain Data
    train_ds からバッチごとにデータを取得します。
2. 前処理:
    必要に応じてデータの前処理を行います。例えば、特徴量のスケーリングやエンコーディングなど。
3.フォワードパス:

バッチ内のデータをモデルに入力し、予測を行います。
4. 損失計算:

モデルの予測と実際のラベル（Survived）との誤差（損失）を計算します。
5. バックプロパゲーション:

損失を基にモデルのパラメータを更新します。
6. 次のバッチ:

次のバッチを取得し、ステップ3からステップ5を繰り返します。
7. エポックの完了:

8. 全データが処理されると1エポックが完了します。これを指定したエポック数だけ繰り返します。
"""