from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import json
import pandas as pd


import boto3

client = boto3.client("s3")
class OriginLensModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(
      self,
      user_model: tf.keras.Model,
      origin_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_model = user_model
    self.origin_model = origin_model

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.

    user_embeddings = self.user_model(features["id"])
    origin_embeddings = self.origin_model(features["Destination"])

    return self.task(user_embeddings, origin_embeddings)

def hello(event, context):
    try:
        data = client.get_object(Bucket='travel-listing-from-makemytrip', Key='makemytrip_travel_data_clean_csv.csv')
        pandasDF = pd.read_csv(data.get("Body"))
        # Features of all the available flight data.
        training_data = tf.data.Dataset.from_tensor_slices(dict(pandasDF))
        origins = tf.data.Dataset.from_tensor_slices(dict(pandasDF.drop_duplicates(subset=['Destination'], keep='last')))
        
        # Select the basic features.
        training_data = training_data.map(lambda x: {
            "Destination": x["Destination"],
            "Origin": x["Origin"],
            "id": x["id"]
        })
        origins = origins.map(lambda x: x["Destination"])
        user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
        user_ids_vocabulary.adapt(training_data.map(lambda x: x["id"]))

        origins_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
        origins_vocabulary.adapt(origins)

        # # Define user and movie models.
        user_model = tf.keras.Sequential([
            user_ids_vocabulary,
            tf.keras.layers.Embedding(user_ids_vocabulary.vocabulary_size(), 64)
        ])
        origin_model = tf.keras.Sequential([
            origins_vocabulary,
            tf.keras.layers.Embedding(origins_vocabulary.vocabulary_size(), 64)
        ])

        # # Define your objectives.
        task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
            origins.batch(128).map(origin_model).cache()
        ))
        # # Create a retrieval model.
        model = OriginLensModel(user_model, origin_model, task)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

        # # Train for 3 epochs.
        model.fit(training_data.batch(4096), epochs=3)

        # # Use brute-force search to set up retrieval using the trained representations.
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        index.index_from_dataset(
            origins.batch(100).map(lambda origin: (origin, model.origin_model(origin))))

        # # Get some recommendations.
        _, recommendedDestinations = index(np.array(["42"]))
        body = {
            "data": f"Top 3 recommendations for user 42: {recommendedDestinations[0, :3]}",
            "input": event,
        }

        response = {"statusCode": 200, "body": json.dumps(body)}

        return response
    except:
        response = {"statusCode": 500, "body": "Error"}

        return response