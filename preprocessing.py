from typing import Tuple

import pandas as pd


def drop_vague_elements(df: pd.DataFrame, min_ratings: int) -> pd.DataFrame:
    """
    Iteratively look for items and users having too few ratings and drop them.
    """
    initial = df
    df = df.copy()
    iteration = 0
    while True:
        print(f"iteration {iteration}")
        iteration += 1
        ratings_per_user = df.groupby('reviewerID').size()
        vague_users = ratings_per_user[ratings_per_user < min_ratings].index.values
        print(f'# of vague users: {len(vague_users)}')

        df = df[~df.reviewerID.isin(vague_users)]

        ratings_per_item = df.groupby('asin').size()
        vague_items = ratings_per_item[ratings_per_item < min_ratings].index.values
        print(f'# of vague items: {len(vague_items)}')

        df = df[~df.asin.isin(vague_items)]

        if len(vague_users) == 0 and len(vague_items) == 0:
            print("what's left:")
            print(f"- {len(df) / len(initial):.1%} of ratings")
            print(f"- {df.asin.nunique() / initial.asin.nunique():.1%} of unique items")
            print(f"- {df.reviewerID.nunique() / initial.reviewerID.nunique():.1%} of unique users")
            return df


def index_items(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enumerate items of each user in historical order.
    Store the indices in the "item_index" column of the output dataframe.
    """
    df = df.sort_values(['reviewerID', 'unixReviewTime'])
    item_indices = []
    index = 0
    prev = None
    for user in df.reviewerID.values:
        if user != prev:
            index = 0
            prev = user
        item_indices.append(index)
        index += 1
    df['item_index'] = item_indices
    return df


def split_by_user(df: pd.DataFrame, train_ratings_num: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Put the first train_ratings_num items of each user to the train split
    and put the rest to the test split.
    """
    df = index_items(df)
    train = df[df.item_index < train_ratings_num].drop(columns=['item_index'])
    test = df[df.item_index >= train_ratings_num].drop(columns=['item_index'])
    return train, test
