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
        tf.keras.layers.StringLookup(vocabulary=unique_user_ids,mask_token=None),
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension),
        tf.keras.layers.Reshape([-1])
    ])

    self.destination_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=unique_destinations,mask_token=None),
        tf.keras.layers.Embedding(len(unique_destinations) + 1, embedding_dimension),
        tf.keras.layers.Reshape([-1])
    ])

    self.travel_date_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=unique_travel_dates, mask_token=None),
        tf.keras.layers.Embedding(len(unique_travel_dates) + 1, embedding_dimension),
        tf.keras.layers.Reshape([-1])
    ])

    self.origin_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=unique_origins, mask_token=None),
        tf.keras.layers.Embedding(len(unique_origins) + 1, embedding_dimension),
        tf.keras.layers.Reshape([-1])
    ])

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
    
    self.destination_retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=destinations.batch(128).map(self.destination_model)
        )
    )
    # self.origin_retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
    #     metrics=tfrs.metrics.FactorizedTopK(
    #         origins.batch(128).map(self.origin_model)
    #     )
    # )
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
    user_embedding = self.user_model(features["id"])
    # And pick out the destination, travel date, and origin features and pass them into their models.
    destination_embedding = self.destination_model(features["Destination"])
    travel_date_embedding = self.travel_date_model(features["Travel_Date"])
    origin_embedding = self.origin_model(features["Origin"])

    return  user_embedding, destination_embedding, travel_date_embedding, origin_embedding, self.rating_model(tf.concat([user_embedding, destination_embedding], axis=1))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.
    fare_price = features.pop("Fare_Price")
    user_embedding, destination_embedding, travel_date_embedding, origin_embedding, fare_price_predictions = self(features)

    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=fare_price,
        predictions=fare_price_predictions,
    )
    destination_retrieval_loss = self.destination_retrieval_task(user_embedding,destination_embedding)
    # origin_retrieval_loss = self.origin_retrieval_task(destination_embedding,origin_embedding)
    # travel_date_retrieval_loss = self.travel_date_retrieval_task(origin_embedding,travel_date_embedding)
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
origins = tf.data.Dataset.from_tensor_slices(dict(pandasDF.drop_duplicates(subset="Origin")))    

# Select the basic features.
training_data = training_data.map(lambda x: {
    "id": x["id"],
    "Destination": x["Destination"],
    "Origin": x["Origin"],
    "Travel_Date": x["Travel_Date"],
    "Fare_Price": x["Fare_Price"]      
})

destinations = destinations.map(lambda x: x["Destination"])
origins = origins.map(lambda x: x["Origin"])
travel_dates = travel_dates.map(lambda x: x["Travel_Date"])

# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = training_data.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

ids = training_data.batch(1_000_000).map(lambda x: x["id"])
destination_names = destinations.batch(1_000)
travel_dates_batch = travel_dates.batch(1_000)
origin_names = origins.batch(1_000)

unique_user_ids = np.unique(np.concatenate(list(ids)))
unique_destinations = np.unique(np.concatenate(list(destination_names)))
unique_travel_dates = np.unique(np.concatenate(list(travel_dates_batch)))
unique_origins = np.unique(np.concatenate(list(origin_names)))

# Let's now train a model that assigns positive weights to both retrieval and ranking tasks.
model = ItineraryModel(rating_weight=0, retrieval_weight=1.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1), loss=tf.keras.losses.CategoricalCrossentropy())
       
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

def getItineraries(event, context):
    try:
        model.fit(cached_train, epochs=15)
        # metrics = model.evaluate(cached_test, return_dict=True)

        # print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
        # print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")

         # # Use brute-force search to set up retrieval using the trained representations.
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        index.index_from_dataset(
            destinations.batch(100).map(lambda destination: (destination, model.destination_model(destination))))
        _, recommendedDestinations = index(np.array(["00021700f41a71382d3f5f1d87ed3e72"]))
        itinerary = {}
        
        for idx, destination in enumerate(recommendedDestinations[0, :3]):
            index.index_from_dataset(
            travel_dates.batch(100).map(lambda travelDate: (travelDate, model.travel_date_model(travelDate))))
            # Get some travel date recommendations.
            _, recommendedTravelDates = index(np.array([str(destination)]))
            topDates = recommendedTravelDates[0,:3]
            itinerary[f"itinerary-{idx}"] = {
                 "id": "0021700f41a71382d3f5f1d87ed3e72",
                 "Destination": str(destination),
                 "Travel_Date": str(topDates[idx])
             }

        body = {
            "data": itinerary,
        }

        response = {"statusCode": 200, "body": json.dumps(body)}

        return response
    except Exception as e:
        tf.print(e)
        response = {"statusCode": 500, "body": "Error occurred"}

        return response