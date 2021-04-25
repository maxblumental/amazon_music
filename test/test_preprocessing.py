import numpy as np
import pandas as pd
import pytest

from preprocessing import drop_vague_elements, split_by_user, index_items


@pytest.fixture
def ratings() -> pd.DataFrame:
    return pd.DataFrame([
        {'reviewerID': '1', 'asin': '1', 'overall': 5},
        {'reviewerID': '1', 'asin': '2', 'overall': 4},
        {'reviewerID': '2', 'asin': '1', 'overall': 3},
        {'reviewerID': '2', 'asin': '2', 'overall': 5},
        {'reviewerID': '7', 'asin': '1', 'overall': 5},
        {'reviewerID': '7', 'asin': '13', 'overall': 4},
        {'reviewerID': '7', 'asin': '15', 'overall': 4},
    ])


def test_drop_vague_elements(ratings):
    result = drop_vague_elements(ratings, min_ratings=2)

    assert len(result) == 4
    assert 7 not in result.reviewerID.unique()
    assert 13 not in result.asin.unique()
    assert 15 not in result.asin.unique()


@pytest.fixture
def ratings_with_time() -> pd.DataFrame:
    return pd.DataFrame([
        {'reviewerID': '1', 'asin': '1', 'overall': 5, 'unixReviewTime': 1},
        {'reviewerID': '1', 'asin': '2', 'overall': 5, 'unixReviewTime': 2},
        {'reviewerID': '1', 'asin': '3', 'overall': 4, 'unixReviewTime': 3},
        {'reviewerID': '2', 'asin': '4', 'overall': 5, 'unixReviewTime': 1},
        {'reviewerID': '2', 'asin': '5', 'overall': 3, 'unixReviewTime': 2},
        {'reviewerID': '3', 'asin': '6', 'overall': 5, 'unixReviewTime': 1},
    ])


def test_index_items(ratings_with_time):
    ratings = ratings_with_time.sample(frac=1, random_state=0)
    result = index_items(ratings)

    indices = result.sort_values(['reviewerID', 'unixReviewTime']).item_index.values
    assert np.array_equal(indices, [0, 1, 2, 0, 1, 0])


def test_split_by_user(ratings_with_time):
    ratings = ratings_with_time.sample(frac=1, random_state=0)
    train, test = split_by_user(ratings, train_ratings_num=1)

    assert np.array_equal(ratings.columns, train.columns)
    assert np.array_equal(ratings.columns, test.columns)

    assert len(train) == 3
    assert len(test) == 3
    assert sorted(train.reviewerID.unique()) == ['1', '2', '3']
    assert sorted(test.reviewerID.unique()) == ['1', '2']
    assert sorted(train.asin.unique()) == ['1', '4', '6']
    assert sorted(test.asin.unique()) == ['2', '3', '5']
