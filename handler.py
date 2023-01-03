from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import json
import pandas as pd


import boto3

client = boto3.client("s3")
class Model(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(
      self,
      user_model: tf.keras.Model,
      destination_model: tf.keras.Model,
      traveldates_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_model = user_model
    self.destination_model = destination_model
    self.traveldates_model = traveldates_model

    # Set up a retrieval task.
    self.task = task
  # We can define a TFRS model by inheriting from tfrs.Model and implementing the compute_loss method
  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.
    user_embeddings = self.user_model(features["id"])
    destination_embeddings = self.destination_model(features["Destination"])
    traveldates_embeddings = self.traveldates_model(features["Travel_Date"])

    return self.task(user_embeddings, destination_embeddings, traveldates_embeddings)

def hello(event, context):
    try:
        # read from the CSV hosted in the AWS S3 bucket
        data = client.get_object(Bucket='travel-listing-from-makemytrip', Key='makemytrip_travel_data_clean_csv.csv')
        pandasDF = pd.read_csv(data.get("Body"))
        # Features of all the available flight data.
        training_data = tf.data.Dataset.from_tensor_slices(dict(pandasDF))
        destinations = tf.data.Dataset.from_tensor_slices(dict(pandasDF.drop_duplicates(subset=['Destination'], keep="last")))
        travelDates = tf.data.Dataset.from_tensor_slices(dict(pandasDF.drop_duplicates(subset=['Travel_Date'], keep="last")))

        # Select the basic features.    
        # These are the fields we are going to recommend 
        recommendedDestinations = destinations.map(lambda x: x["Destination"])
        recommendedTravelDates = travelDates.map(lambda x: x["Travel_Date"]) 

        # Build vocabularies to convert user ids and movie titles into integer indices for embedding layers:
        user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
        user_ids_vocabulary.adapt(training_data.map(lambda x: x["id"]))
        destinations_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
        destinations_vocabulary.adapt(recommendedDestinations)
        travelDates_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
        travelDates_vocabulary.adapt(recommendedTravelDates)

        # Define the two models and the retrieval task.
        user_model = tf.keras.Sequential([
            user_ids_vocabulary,
            tf.keras.layers.Embedding(user_ids_vocabulary.vocabulary_size(), 64)
        ])
        destinations_model = tf.keras.Sequential([
            destinations_vocabulary,
            tf.keras.layers.Embedding(destinations_vocabulary.vocabulary_size(), 64)
        ])
        traveldates_model = tf.keras.Sequential([
            travelDates_vocabulary,
            tf.keras.layers.Embedding(travelDates_vocabulary.vocabulary_size(), 64)
        ])

        # Define your objectives.
        task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
            candidates=recommendedDestinations.batch(128).map(destinations_model)
        ))

        # Create the model, train it, and generate predictions:
        # 1. Create a retrieval models.
        model = Model(user_model, destinations_model, traveldates_model, task)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

        # 2. Train for 3 epochs.
        model.fit(training_data.batch(4096), epochs=3)

        # 3. Use brute-force search to set up retrieval using the trained representations.
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        index.index_from_dataset(recommendedDestinations.batch(100).map(lambda destination: (destination, model.destination_model(destination))))

        # 4. Get some recommendations.
        _, recommendedDestinations = index(np.array(["c50f8f0eba9e2d55bc9fcf94f0f9e8cb"]))
        
        body = {
            "data": f"Top 3 recommendations for user c50f8f0eba9e2d55bc9fcf94f0f9e8cb: {recommendedDestinations[0, :3]}",
            "input": event,
        }

        response = {"statusCode": 200, "body": json.dumps(body)}

        return response
    except:
        response = {"statusCode": 500, "body": "Error"}

        return response
