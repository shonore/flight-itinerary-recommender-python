#!/usr/bin/env python
from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import json
import pandas as pd

import boto3

class ItineraryModel(tfrs.models.Model):

  def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.

    super().__init__()
   
    embedding_dimension = 32
    
    # User and destination models.
    self.user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=user_ids,mask_token=None),
        tf.keras.layers.Embedding(len(user_ids) + 1, embedding_dimension),
        tf.keras.layers.Reshape([-1,])
    ])

    self.destination_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=destinations,mask_token=None),
        tf.keras.layers.Embedding(len(destinations) + 1, embedding_dimension),
        tf.keras.layers.Reshape([-1,])
    ])

    # self.travel_date_model = tf.keras.Sequential([
    #     tf.keras.layers.StringLookup(vocabulary=unique_travel_dates, mask_token=None),
    #     tf.keras.layers.Embedding(len(unique_travel_dates) + 1, embedding_dimension),
    #     tf.keras.layers.Reshape([-1,])
    # ])

    self.rating_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
   
    # The tasks.
    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            destinations.batch(128).map(self.destination_model)
        )
    )
    # self.travel_date_retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
    #     metrics=tfrs.metrics.FactorizedTopK(
    #         travel_dates.batch(128).map(self.travel_date_model)
    #     )
    # )
    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["id"])
    # And pick out the destination features and pass them into the destination model.
    destination_embeddings = self.destination_model(features["Destination"])
    #travel_date_embeddings = self.travel_date_model(features["Travel_Date"])
    return (
        user_embeddings,
        destination_embeddings,
        self.rating_model(
            tf.concat([user_embeddings, destination_embeddings], axis=1)
        ),
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.
    fare_price = features.pop("Fare_Price")
    user_embeddings, destination_embeddings, fare_price_predictions = self(features)

    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=fare_price,
        predictions=fare_price_predictions,
    )
    destination_retrieval_loss = self.retrieval_task(user_embeddings, destination_embeddings)
    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss
            + self.retrieval_weight * destination_retrieval_loss)


client = boto3.client("s3")
data = client.get_object(Bucket='travel-listing-from-makemytrip', Key='makemytrip_travel_data_clean_csv.csv')
pandasDF = pd.read_csv(data.get("Body"))
# Features of all the available flight data.
training_data = tf.data.Dataset.from_tensor_slices(dict(pandasDF))
destinations = tf.data.Dataset.from_tensor_slices(dict(pandasDF.drop_duplicates(subset="Destination")))
travel_dates = tf.data.Dataset.from_tensor_slices(dict(pandasDF.drop_duplicates(subset="Travel_Date")))
       
# Select the basic features.
training_data = training_data.map(lambda x: {
    "id": x["id"],
    "Destination": x["Destination"],
    "Origin": x["Origin"],
    "Travel_Date": x["Travel_Date"],
    "Fare_Price": x["Fare_Price"]      
})

destinations = destinations.map(lambda x: x["Destination"])
travel_dates = travel_dates.map(lambda x: x["Travel_Date"])
user_ids = training_data.map(lambda x: x["Travel_Date"])
# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = training_data.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# Let's now train a model that assigns positive weights to both retrieval and ranking tasks.
model = ItineraryModel(rating_weight=1.0, retrieval_weight=1.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
       
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()
# cached_train = cached_train.map(lambda x: {
#             "id": tf.reshape(x["id"], [1,1]),
#             "Destination": tf.reshape(x["Destination"], [1,1]),
#             "Origin": tf.reshape(x["Origin"], [1,1]),
#             "Travel_Date": tf.reshape(x["Travel_Date"], [1,1]),
#             "Fare_Price": tf.reshape(x["Fare_Price"], [1,1]),
# }).cache()

def getItineraries(event, context):
    try:
        model.fit(cached_train, epochs=3, verbose=1)

         # # Use brute-force search to set up retrieval using the trained representations.
        test_destinations = cached_test.map(lambda x: x["Destination"])
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        index.index_from_dataset(
            test_destinations.batch(100).map(lambda destination: (destination, model.destination_model(destination))))

        # Get some recommendations.
        _, recommendedDestinations = index(np.array(["00021700f41a71382d3f5f1d87ed3e72"]))
        tf.print(recommendedDestinations[0,:3])
        test_ratings = {}
        topDestinations = recommendedDestinations[0, :3]
        for destination in topDestinations:
            test_ratings[destination] = model({
                "id": np.array(["0021700f41a71382d3f5f1d87ed3e72"]),
                "Destination": np.array([destination])
            })

        tf.print("Travel Dates:")
        for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
            tf.print(f"{title}: {score}")

        body = {
            "data": f"wip",
            "input": event,
        }

        response = {"statusCode": 200, "body": json.dumps(body)}

        return response
    except Exception as e:
        tf.print(e)
        response = {"statusCode": 500, "body": "Error occurred"}

        return response