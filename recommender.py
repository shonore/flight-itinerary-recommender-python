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

  def __init__(self, unique_destinations, unique_user_ids, destinations, rating_weight: float, retrieval_weight: float) -> None:
    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.

    super().__init__()

    embedding_dimension = 32
    # User and destination models.
    self.destination_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_destinations, mask_token=None),
      tf.keras.layers.Embedding(len(unique_destinations) + 1, embedding_dimension)
    ])
    self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # A small model to take in user and destination embeddings and predict ratings.
    # We can make this as complicated as we want as long as we output a scalar
    # as our prediction.
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
            candidates=destinations.batch(128).map(self.destination_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight


  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["id"])
    # And pick out the destination features and pass them into the destination model.
    destination_embeddings = self.destination_model(features["Destination"])

    return (
        user_embeddings,
        destination_embeddings,
        # We apply the multi-layered rating model to a concatentation of
        # user and destination embeddings.
        self.rating_model(
            tf.concat([user_embeddings, destination_embeddings], axis=1)
        ),
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    tavel_dates = features.pop("Origin")
    user_embeddings, destination_embeddings, rating_predictions = self(features)
    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=tavel_dates,
        predictions=rating_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, destination_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss
            + self.retrieval_weight * retrieval_loss)

def getItineraries(event, context):
    client = boto3.client("s3")
    try:
        data = client.get_object(Bucket='travel-listing-from-makemytrip', Key='makemytrip_travel_data_clean_csv.csv')
        pandasDF = pd.read_csv(data.get("Body"))
        tf.print(pandasDF.dtypes)
        # Features of all the available flight data.
        training_data = tf.data.Dataset.from_tensor_slices(dict(pandasDF))
        destinations = tf.data.Dataset.from_tensor_slices(dict(pandasDF.drop_duplicates(subset=['Destination'])))
       
        # Select the basic features.
        training_data = training_data.map(lambda x: {
            "id": x["id"],
            "Destination": x["Destination"],
            "Origin": x["Origin"],
            "Travel_Date": x["Travel_Date"],
            "Fare_Price": x["Fare_Price"]
        })
        
        destinations = destinations.map(lambda x: x["Destination"])
        # Randomly shuffle data and split between train and test.
        tf.random.set_seed(42)
        shuffled = training_data.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

        train = shuffled.take(80_000)
        test = shuffled.skip(80_000).take(20_000)

        destination_names = destinations.batch(1_000)
        ids = training_data.batch(1_000_000).map(lambda x: x["id"])

        unique_destinations = np.unique(np.concatenate(list(destination_names)))
        unique_user_ids = np.unique(np.concatenate(list(ids)))

        # Let's now train a model that assigns positive weights to both retrieval and ranking tasks.
        model = ItineraryModel(unique_destinations, unique_user_ids, destinations, rating_weight=1.0, retrieval_weight=1.0)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

        cached_train = train.shuffle(100_000).batch(8192).cache()
        cached_test = test.batch(4096).cache()

        model.fit(cached_train, epochs=3)
        metrics = model.evaluate(cached_test, return_dict=True)

        print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
        print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")

        body = {
            "data": f"wip",
            "input": event,
        }

        response = {"statusCode": 200, "body": json.dumps(body)}

        return response
    except:
        response = {"statusCode": 500, "body": "Error"}

        return response