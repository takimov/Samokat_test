from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union
import pandas as pd
from etna.transforms import TimeSeriesImputerTransform
from etna.datasets import TSDataset
from etna.transforms.base import IrreversibleTransform
from enum import Enum
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from etna.datasets.utils import determine_freq
from etna.datasets.utils import determine_num_steps
from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.transforms.base import OneSegmentTransform
from etna.transforms.base import ReversiblePerSegmentWrapper
class CustomLagTransform(IrreversibleTransform):
    """Generates series of lags from given dataframe.

    Notes
    -----
    Types of shifted variables could change due to applying :py:meth:`pandas.DataFrame.shift`.
    """

    def __init__(self, in_column: str, lags: Union[Dict[str, List[int]], int], out_column: Optional[str] = None):
        """Create instance of LagTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        lags:
            int value or list of values for lags computation; if int, generate range of lags from 1 to given value
        out_column:
            base for the name of created columns;

            * if set the final name is '{out_column}_{lag_number}';

            * if don't set, name will be ``transform.__repr__()``,
              repr will be made for transform that creates exactly this column

        Raises
        ------
        ValueError:
            if lags value contains non-positive values
        """
        super().__init__(required_features=[in_column])
        if isinstance(lags, int):
            if lags < 1:
                raise ValueError(f"{type(self).__name__} works only with positive lags values, {lags} given")
            self.lags = list(range(1, lags + 1))
        else:
            self.lags = lags

        self.in_column = in_column
        self.out_column = out_column

    def _get_column_name(self, lag: int) -> str:
        if self.out_column is None:
            temp_transform = CustomLagTransform(in_column=self.in_column, out_column=self.out_column, lags=[lag])
            return repr(temp_transform)
        else:
            return f"{self.out_column}_{lag}"

    def _fit(self, df: pd.DataFrame) -> "CustomLagTransform":
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: LagTransform
        """
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lags to the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.Dataframe
            transformed dataframe
        """
        result = df

        segments = sorted(set(df.columns.get_level_values("segment")))
        all_transformed_features = []
        features = df.loc[:, pd.IndexSlice[:, self.in_column]]
        for segment in segments:
            for lag in self.lags[segment]:
                column_name = self._get_column_name(lag)
                # this could lead to type changes due to introduction of NaNs
                transformed_features = features[(segment,'target')].shift(lag)
                transformed_features.rename((segment,column_name),inplace=True)
                all_transformed_features.append(transformed_features)
        result = pd.concat([result] + all_transformed_features, axis=1)
        result = result.sort_index(axis=1)
        return result

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return [self._get_column_name(lag) for lag in self.lags]

_DEFAULT_FREQ = object()

class DeseasonalModel(str, Enum):
    """Enum for different types of deseasonality model."""

    additive = "additive"
    multiplicative = "multiplicative"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} types allowed."
        )

class CustomSeasonal(IrreversibleTransform):
    """Generates series of lags from given dataframe.

    Notes
    -----
    Types of shifted variables could change due to applying :py:meth:`pandas.DataFrame.shift`.
    """

    def __init__(self, in_column: str, period:Dict[str, int], lag: int, out_column: Optional[str] = None, model: str = DeseasonalModel.additive):
        """Create instance of LagTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        lags:
            int value or list of values for lags computation; if int, generate range of lags from 1 to given value
        out_column:
            base for the name of created columns;

            * if set the final name is '{out_column}_{lag_number}';

            * if don't set, name will be ``transform.__repr__()``,
              repr will be made for transform that creates exactly this column

        Raises
        ------
        ValueError:
            if lags value contains non-positive values
        """
        super().__init__(required_features=[in_column])
        self.lag = lag

        self.in_column = in_column
        self.out_column = out_column
        self.period = period
        self.model = DeseasonalModel(model)
    def _get_column_name(self, lag: int) -> str:
        if self.out_column is None:
            temp_transform = CustomLagTransform(in_column=self.in_column, out_column=self.out_column, lags=[lag])
            return repr(temp_transform)
        else:
            return f"{self.out_column}_{lag}"

    def _fit(self, df: pd.DataFrame) -> "LagTransform":
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: LagTransform
        """
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lags to the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.Dataframe
            transformed dataframe
        """
        result = df
        segments = sorted(set(df.columns.get_level_values("segment"))&set(self.period.keys()))
        all_transformed_features = []
        features = df.loc[:, pd.IndexSlice[:, self.in_column]]
        for segment in segments:
            # this could lead to type changes due to introduction of NaNs
            column_name = self._get_column_name(self.period[segment])
            df_copy = features[(segment,'target')]
            self._freq = determine_freq(df_copy.index)
            df_copy = df_copy.loc[df_copy.first_valid_index() : df_copy.last_valid_index()]
            # df_copy.fillna(df_copy.mean(),inplace=True)
            # if df_copy.isnull().values.any():
            #     raise ValueError("The input column contains NaNs in the middle of the series! Try to use the imputer.")
            _seasonal = seasonal_decompose(
                x=df_copy, model=self.model, period=self.period[segment], filt=None, two_sided=False, extrapolate_trend=0
            ).seasonal
            transformed_features = _seasonal.shift(self.lag)
            transformed_features.rename((segment,column_name),inplace=True)
            all_transformed_features.append(transformed_features)
        result = pd.concat([result] + all_transformed_features, axis=1)
        result = result.sort_index(axis=1)
        return result

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return [self._get_column_name(self.period[segment]) for segment in self.period]