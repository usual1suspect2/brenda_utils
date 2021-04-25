"""
One-hot-encoder class extended to work with pandas DataFrame and produce proper column names
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class OneHotEncoderDF(BaseEstimator, TransformerMixin):
    def __init__(self, columns, prefix=None, prefix_sep="_", drop_first=False):
        """
        One-hot encoder for generating one-hot encoded categorical features
        :param columns:                         a list of column names that require one-hot encoding
        :param prefix:                          either:
                                                 - None (default), original column names will be used as prefixes for
                                                   one-hot encoded columns
                                                 - a string, one prefix that will be used for all columns
                                                 - a list with length equal to the number of columns, one prefix
                                                   for each one-hot encoded column
        :param prefix_sep:                      separator/delimiter to use for appending prefix
        :param drop_first:                      a boolean flag for dropping the first category in each one-hot encoded
                                                column
        """
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        if prefix is None:
            self.prefix = columns
        elif isinstance(prefix, str):
            self.prefix = [prefix for _ in range(len(columns))]
        else:
            if len(prefix) != len(columns):
                raise ValueError(
                    "When `prefix` is not a string, it must be an iterable of equal length with `columns` "
                    "(one prefix for each column). Found: {} != {}".format(len(prefix), len(columns))
                )
            self.prefix = prefix
        self.prefix_sep = prefix_sep
        self.columns = columns
        self.drop_first = drop_first

    def fit(self, x, y=None):
        """
        Fit OneHotEncoder to data
        :param x:                               input DataFrame with columns required to determine the categories of
                                                each one-hot encoded feature
        :param y:                               ignored
        :return:                                self
        """
        for column in self.columns:
            if column not in x.columns:
                raise ValueError(
                    "Column `{}` for one-hot-encoding wasn't found in the input DataFrame".format(column)
                )
            if x[column].nunique() < 2:
                raise ValueError(
                    "Trying to one-hot-encode column with a single unique value"
                )
        self.encoder.fit(x[self.columns], y)
        return self

    def transform(self, x):
        """
        Transforms input DataFrame applying one-hot encoding for specified columns
        :param x:                               DataFrame with columns to encode
        :return:                                Transformed DataFrame with all specified features one-hot encoded.
                                                Original columns are removed from the DataFrame during transformation
        """
        for column in self.columns:
            if column not in x.columns:
                raise ValueError(
                    "Column `{}` for one-hot-encoding wasn't found in the input DataFrame".format(column)
                )
        transformed_array = self.encoder.transform(x[self.columns]).toarray()
        transformed_col_names = []
        columns_to_drop = []
        for i, column in enumerate(self.columns):
            prefix = self.prefix[i]
            level_names = [self.prefix_sep.join([prefix, val]) for val in self.encoder.categories_[i]]
            transformed_col_names.extend(level_names)
            if self.drop_first:
                columns_to_drop.append(level_names[0])
        transformed = pd.DataFrame(transformed_array, index=x.index, columns=transformed_col_names, dtype=np.int8)
        transformed = transformed.drop(columns=columns_to_drop)
        transformed = pd.concat((x.drop(columns=self.columns), transformed), axis=1)
        return transformed
