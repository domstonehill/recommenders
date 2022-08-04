import pandas as pd
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise import SVD as Algorithm
from surprise.model_selection import GridSearchCV


# CHANGE THIS TO MATCH WHERE YOU DOWNLOADED THE BOOK DATASET
FILEPATH = '/home/dominik/cifsmnt/book_recommendations/Ratings.csv'


def load_data():
    '''
    Loads in data
    :return:
    '''

    data = pd.read_csv(FILEPATH)
    data.columns = ["user", "item", "rating"]

    return data


def build_dataset(df: pd.DataFrame):
    '''
    Builds a Surprise Dataset from the input df
    :return:
    '''

    reader = Reader(rating_scale=(df.rating.min(), df.rating.max()))
    ds = Dataset.load_from_df(df, reader)

    return ds


if __name__ == '__main__':
    data = load_data()
    print(data)
    print(data.nunique(axis=0))
    ds = build_dataset(data)
    print(ds.raw_ratings[:5])

    param_grid = {
        "n_epochs": [5, 10],
        "lr_all": [2e-3, 5e-3],
        "reg_all": [0.4, 0.6]
    }

    gs = GridSearchCV(Algorithm, param_grid, measures=["rmse", "mae"], cv=3, n_jobs=4)

    gs.fit(ds)

    # Best RMSE Score
    print("Best Score:")
    print(gs.best_score['rmse'])
    print()
    # Combination of parameters that give best score
    print("Best Parameters:")
    print(gs.best_params['rmse'])
