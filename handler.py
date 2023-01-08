from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import json
import pandas as pd


import boto3

client = boto3.client("s3")
class DestinationsModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model(
        (features["id"], features["Destination"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("Travel_Date")

    rating_predictions = self(features)

    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)
class RankingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    embedding_dimension = 32

    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=self.user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(self.user_ids) + 1, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.origin_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=self.origins, mask_token=None),
      tf.keras.layers.Embedding(len(self.origins) + 1, embedding_dimension)
    ])

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
  ])

  def call(self, inputs):

    user_id, destination = inputs

    user_embedding = self.user_embeddings(user_id)
    origin_embedding = self.origin_embeddings(destination)

    return self.ratings(tf.concat([user_embedding, origin_embedding], axis=1))
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

def poc(event, context):
    try:
        data = client.get_object(Bucket='travel-listing-from-makemytrip', Key='makemytrip_travel_data_clean_csv.csv')
        pandasDF = pd.read_csv(data.get("Body"))
        # Features of all the available flight data.
        training_data = tf.data.Dataset.from_tensor_slices(dict(pandasDF.drop_duplicates(subset=['id'])))
        origins = tf.data.Dataset.from_tensor_slices(dict(pandasDF.drop_duplicates(subset=['Destination'], keep='last')))
        
        # Select the basic features.
        training_data = training_data.map(lambda x: {
            "Destination": x["Destination"],
            "Origin": x["Origin"],
            "id": x["id"],
            "Travel_Date": x["Travel_Date"],
            "Fare_Price": x["Fare_Price"]
        })
        origins = origins.map(lambda x: x["Destination"])

        # tf.random.set_seed(42)
        # shuffled = training_data.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

        # train = shuffled.take(80_000)
        # test = shuffled.skip(80_000).take(20_000)

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
        task1 = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
            origins.batch(128).map(origin_model).cache()
        ))
        # # Create a retrieval model.
        model1 = OriginLensModel(user_model, origin_model, task1)
        model1.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

        # # Train for 3 epochs.
        model1.fit(training_data.batch(4096), epochs=5)

        # # Use brute-force search to set up retrieval using the trained representations.
        index = tfrs.layers.factorized_top_k.BruteForce(model1.user_model)
        index.index_from_dataset(
            origins.batch(100).map(lambda origin: (origin, model1.origin_model(origin))))

        # Get some recommendations.
        _, recommendedDestinations = index(np.array(["00021700f41a71382d3f5f1d87ed3e72"]))
        #tf.print("Top 3 recommendations for user 00021700f41a71382d3f5f1d87ed3e72:" + recommendedDestinations[0, :3])
        #Ranking
        # model2 = DestinationsModel(user_model, origin_model)
        # model2.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

        # cached_train = train.shuffle(100_000).batch(8192).cache()
        # cached_test = test.batch(4096).cache()

        # model2.fit(cached_train, epochs=3)
        # test_ratings = {}
        # test_destinations = recommendedDestinations[0, :3]
        # for destination in test_destinations:
        #     test_ratings[destination] = model2({
        #         "id": np.array(["00021700f41a71382d3f5f1d87ed3e72"]),
        #         "Destination": np.array([destination])
        # })

        # tf.print("Travel Dates:")
        # for destination, travel_date in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
        #     tf.print(f"{destination}: {travel_date}")
        body = {
            "data": f"Top 3 recommendations for user 00021700f41a71382d3f5f1d87ed3e72: {recommendedDestinations[0, :3]}.",
            "input": event,
        }

        response = {"statusCode": 200, "body": json.dumps(body)}

        return response
    except:
        response = {"statusCode": 500, "body": "Error"}

        return response