import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

import matplotlib.pyplot as plt


class MovieLensModel(tfrs.Model):
    '''
    A model for making watch suggestions for Movies from the MovieLens Dataset
    '''

    def __init__(
            self,
            user_model: tf.keras.Model,
            movie_model: tf.keras.Model,
            task: tfrs.tasks.Retrieval
    ):
        super(MovieLensModel, self).__init__()

        # Set up user and movie representations
        self.user_model = user_model
        self.movie_model = movie_model

        # Set up a retrieval task
        self.task = task

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        '''
        Computes the loss during training

        :param inputs: The inputs from a training step
        :param training: Boolean indicating whether this is for a training step or inference
        :return: the loss
        '''

        user_embeddings = self.user_model(inputs["user_id"])
        moview_embeddings = self.movie_model(inputs["movie_title"])

        return self.task(user_embeddings, moview_embeddings)


def load_datasets():
    '''
    Loads the MovieLens 100k dataset and preprocesses it (but it doesn't encode the words)
    :return:
    '''

    # Ratings data
    ratings = tfds.load('movielens/100k-ratings', split="train")
    for d in ratings.take(1):
        print(d)
    # Features of all the available movies.
    movies = tfds.load('movielens/100k-movies', split="train")

    # Select the basic features.
    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"]
    })
    movies = movies.map(lambda x: x["movie_title"])

    return ratings, movies


def build_user_model(ratings: tf.data.Dataset):
    '''
    Builds the user_model for use in the MovieLens recommender
    :param ratings: The ratings dataset
    :return: the user_model
    '''

    # Build vocabulary to convert user ids into integer indices for embedding layers
    user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))

    # Define the user model
    user_model = tf.keras.Sequential([
        user_ids_vocabulary,
        tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
    ])

    return user_model


def build_movie_model(movies: tf.data.Dataset):
    '''
    Builds the movie_model for use in the MovieLens recommender
    :param movies: The movies dataset
    :return: the movies_model
    '''

    # Build vocabulary to convert movie ids into integer indices for embedding layers
    movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    movie_titles_vocabulary.adapt(movies)

    # Define the movie model
    movie_model = tf.keras.Sequential([
        movie_titles_vocabulary,
        tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64)
    ])

    return movie_model


def plot_results_from_history(history):
    # Plot results
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex='all')

    ax1.plot(history['factorized_top_k/top_100_categorical_accuracy'])
    ax1.plot(history['factorized_top_k/top_50_categorical_accuracy'])
    ax1.plot(history['factorized_top_k/top_10_categorical_accuracy'])
    ax1.plot(history['factorized_top_k/top_5_categorical_accuracy'])
    ax1.plot(history['factorized_top_k/top_1_categorical_accuracy'])
    ax1.legend(['Top 100', 'Top 50', 'Top 10', 'Top 5', 'Top 1'])
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_title("Accuracy Metrics vs. Epoch")

    ax2.plot(history["total_loss"])
    ax2.set_title("Total Loss vs. Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")

    plt.show()


if __name__ == '__main__':

    # Load ratings and movies datasets
    ratings, movies = load_datasets()

    # Build the user model
    user_model = build_user_model(ratings)

    # Build the movie model
    movie_model = build_movie_model(movies)

    # Define the task
    task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(movies.batch(128).map(movie_model))
    )

    # Create the retrieval model
    model = MovieLensModel(user_model, movie_model, task)
    model.compile(
        optimizer=tf.keras.optimizers.Adagrad(0.5)
    )

    # Train for 3 epochs
    history = model.fit(ratings.batch(4096), epochs=15)

    # Use brute-force search to set up retrieval using the trained representations
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        movies.batch(100).map(lambda title: (title, model.movie_model(title)))
    )

    # Get some recommendations
    user_id = 43
    _, titles = index(np.array([f"{user_id}"]))
    print(f"Top 3 recommendations for user {user_id}: {titles[0, :3]}")

    # Plot the training results
    plot_results_from_history(history.history)
