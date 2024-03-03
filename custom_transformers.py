import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, PowerTransformer, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    Select columns from pandas dataframe by specifying a list of column names
    '''
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return np.array(X.loc[:, self.attribute_names].values)
        elif isinstance(X, np.ndarray):
            return X
            # else:
            #     raise ValueError("Number of attribute_names must match the number of columns in the NumPy array.")
        else:
            raise TypeError("Input must be a pandas DataFrame or NumPy array.")

class NullsColRemoval(BaseEstimator, TransformerMixin):
    '''
    Transformer to remove columns with high percentage of null values
    '''

    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def fit(self, X, y=None):
        if X.dtype == 'object':
            self.null_bool = np.sum(pd.isnull(X), axis=0) / X.shape[0] >= self.threshold
        else:
            self.null_bool = np.sum(np.isnan(X), axis=0) / X.shape[0] >= self.threshold
        print("hello")
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return np.array(X.T[~self.null_bool].T)

    def get_feature_names(self, input_features=None):
        return input_features[~self.null_bool]

    def get_feature_names_drop(self, input_features=None):
        return input_features[self.null_bool]

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    '''
    Encode the categorical variables either as Label or Hot Encode.
    '''

    def __init__(self, OneHot=True, sparse=False, drop_ix = [], optional_drop_ix = None):
        self.OneHot = OneHot
        self.sparse = sparse
        self.OneHot_obj = OneHotEncoder(sparse=self.sparse)
        self.drop_ix = drop_ix
        self.optional_drop_ix = optional_drop_ix

    def fit(self, X, y=None):
        if self.OneHot:
            self.OneHot_obj.fit(X, y=None)
        return self

    def transform(self,X):
        '''
        Transforms  X using either ΟneHot or LabelEncoder().
        '''
        if self.OneHot:
            #self.one_hot_features = self.OneHot_obj.get_feature_names(self.col_names)
            return self.OneHot_obj.transform(X)

            '''
            Manually specify columns to drop: used for fully correlated columns after
            creating dummy variables.
               '''
            self.n_cols = X.shape[1]
            if self.n_cols != X.shape[1]:
                raise ValueError('Array different n_cols to that fitted')

            self.drop_array = np.zeros(X.shape[1], dtype='bool')

            for i in self.drop_ix:
                self.drop_array[i] = True

            if self.optional_drop_ix:
                for i in self.optional_drop_ix:
                    self.drop_array[i] = True
            return X

        else:
            le = LabelEncoder()
            output = np.apply_along_axis(le.fit_transform, 0, X)
            return output

    def get_feature_names(self, input_features=None):
        if self.OneHot:
            one_hot_features = self.OneHot_obj.get_feature_names(input_features)
            return one_hot_features
        else:
            return input_features

class ZeroVariance(BaseEstimator, TransformerMixin):
    def __init__(self, near_zero=False, freq_cut=95/5, unique_cut=10):
        self.near_zero = near_zero
        self.freq_cut = freq_cut
        self.unique_cut = unique_cut
        self.features_to_keep_ = None

    def fit(self, X, y=None):
        zero_var = np.zeros(X.shape[1], dtype=bool)
        near_zero_var = np.zeros(X.shape[1], dtype=bool)
        n_obs = X.shape[0]

        for i, col in enumerate(np.array(X.T)):
            val_counts = np.unique(col, return_counts=True)
            counts = val_counts[1]
            counts_len = counts.shape[0]
            counts_sort = np.sort(counts)[::-1]

            if counts_len == 1:
                zero_var[i] = True
                near_zero_var[i] = True
                continue

            freq_ratio = counts_sort[0] / counts_sort[1]
            unique_pct = (counts_len / n_obs) * 100

            if (unique_pct < self.unique_cut) and (freq_ratio > self.freq_cut):
                near_zero_var[i] = True

        self.features_to_keep_ = ~near_zero_var if self.near_zero else ~zero_var
        return self

    def transform(self, X, y=None):
        return X.T[self.features_to_keep_].T

    def get_feature_names(self, input_features=None):
        return np.array(input_features)[self.features_to_keep_]


class OptionalSimpleImputer(BaseEstimator, TransformerMixin):
    '''
    Simple wrapper around sklearn.impute SimpleImputer to allow imputation of missing values.
    '''
    def __init__(self, SimpleImpute=True, missing_values=np.nan, strategy='median', copy=True):
        self.SimpleImpute = SimpleImpute
        self.missing_values = missing_values
        self.strategy = strategy
        self.copy = copy
        self.simple_imputed_obj = SimpleImputer(missing_values = self.missing_values,
                                                strategy = self.strategy,
                                                copy = self.copy)

    def fit(self, X, y=None):
        self.simple_imputed_obj.fit(X, y=None)
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        if self.SimpleImpute:
            return self.simple_imputed_obj.transform(X)
        else:
            return X

class FindCorrelation(BaseEstimator, TransformerMixin):
    '''
    Remove pairwise correlations beyond threshold.
    This is not 'exact': it does not recalculate correlation
    after each step, and is therefore less expensive.

    This works similarly to the R caret::findCorrelation function
    with exact = False
    '''
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        '''
        Produce binary array for filtering columns in feature array.
        Remember to transpose the correlation matrix so is
        column major.

        Loop through columns in (n_features,n_features) correlation matrix.
        Determine rows where value is greater than threshold.
        For the candidate pairs, one must be removed. Determine which feature
        has the larger average correlation with all other features and remove it.

        Remember, matrix is symmetric so shift down by one row per column as
        iterate through.
        '''
        self.correlated = np.zeros(X.shape[1], dtype=bool)
        self.corr_mat =  np.corrcoef(X.T)
        abs_corr_mat = np.abs(self.corr_mat)

        for i, col in enumerate(abs_corr_mat.T):
            corr_rows = np.where(col[i+1:] > self.threshold)[0]
            avg_corr = np.mean(col)

            if len(corr_rows) > 0:
                for j in corr_rows:
                    if np.mean(abs_corr_mat.T[:, j]) > avg_corr:
                        self.correlated[j] = True
                    else:
                        self.correlated[i] = True

        return self

    def transform(self, X, y=None):
        '''
        Mask the array with the features flagged for removal
        '''
        return X.T[~self.correlated].T

    def get_feature_names(self, input_features=None):
        return input_features[~self.correlated]

class OptionalExtremeValueHandler(BaseEstimator, TransformerMixin):
    '''
    Transformer to handle extreme values by replacing them with NaN
    based on specified percentiles, and optionally imputing or removing NaN values.
    '''
    def __init__(self, handle_extreme=True, upper_percentile=97.5, lower_percentile=2.5, impute_strategy='mean'):
        self.handle_extreme = handle_extreme
        self.upper_percentile = upper_percentile
        self.lower_percentile = lower_percentile
        self.impute_strategy = impute_strategy
        self.upper_thresholds_ = None
        self.lower_thresholds_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Identify and store dynamic thresholds based on percentiles
        self.upper_thresholds_ = np.nanpercentile(X.values, self.upper_percentile, axis=0)
        self.lower_thresholds_ = np.nanpercentile(X.values, self.lower_percentile, axis=0)

        return self

    def transform(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.handle_extreme:
            # Identify and replace extreme values with NaN
            X.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Replace values exceeding the upper threshold with NaN
            X[X > self.upper_thresholds_] = np.nan

            # Replace values below the lower threshold with NaN
            X[X < self.lower_thresholds_] = np.nan

            # Impute or remove NaN values
            if self.impute_strategy is not None:
                imputer = SimpleImputer(strategy=self.impute_strategy)
                X_imputed = pd.DataFrame(imputer.fit_transform(X))
            else:
                # Alternatively, you can remove rows with NaN values
                X_imputed = X.dropna()

            return np.array(X_imputed)
        else:
            return np.array(X)


class OptionalRobustScaler(BaseEstimator, TransformerMixin):
    '''
    Simple wrapper around sklearn.Preprocessing to allow scaling
    to be toggled as an optional transformation, using RobustScaler
    '''
    def __init__(self, scale=True, quantile_range=(25.0, 75.0), copy=True):
        self.scale = scale
        self.quantile_range = quantile_range
        self.copy = copy
        self.scaler = RobustScaler(
            quantile_range=self.quantile_range,
            copy=self.copy
        )

    def fit(self, X, y=None):
        if self.scale:
            self.scaler.fit(X, y=None)
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        if self.scale:
            return self.scaler.transform(X)
        else:
            return X


class OptionalPowerTransformer(BaseEstimator, TransformerMixin):
    '''
    Simple wrapper around sklearn.Preprocessing PowerTransformer to allow power transform featurewise to
    make data more Gaussian-like.
    '''
    def __init__(self, PowerTransform=True, method='yeo-johnson', standardize=True, copy=True):
        self.PowerTransform = PowerTransform
        self.method = method
        self.standardize = standardize
        self.copy = copy
        self.power_transformed_obj = PowerTransformer(method=self.method, standardize=self.standardize, copy=self.copy)

    def fit(self, X, y=None):
        self.power_transformed_obj.fit(X, y=None)
        return self

    def transform(self, X, y=None):
        if self.PowerTransform:
            return self.power_transformed_obj.transform(X)
        else:
            return X
        
class OptionalPCA(BaseEstimator, TransformerMixin):
    '''
    Simple wrapper around from sklearn.decomposition import PCA to construct uncorrelated features and do feature
    selection
    '''
    def __init__(self, PCATransform=True, copy=True, target_explained_variance=0.99):
        self.PCATransform = PCATransform
        self.copy = copy
        self.target_explained_variance = target_explained_variance
        self.pca_transformed_obj = PCA(copy=self.copy)
        self.num_components_ = None  # To store the number of components

    def fit(self, X, y=None):
        self.pca_transformed_obj.fit(X, y=None)
        self.explained_variance_ratio_ = self.pca_transformed_obj.explained_variance_ratio_
        self.cumulative_explained_variance = self.explained_variance_ratio_.cumsum()
        self.num_components_ = np.argmax(self.cumulative_explained_variance >= self.target_explained_variance) + 1
        return self

    def transform(self, X, y=None):
        if self.PCATransform:
            return self.pca_transformed_obj.transform(X)[:, :self.num_components_]
        else:
            return X

class PipelineChecker(BaseEstimator, TransformerMixin):
    '''
    purpose: to do some error checking,
    e.g. number of columns, sense checks, missing values, extreme values etc
    perhaps store min, max, is negative, is binary, std (flag if gt max + 3 std)

    At the moment only checks to see that number of columns in train the same
    as anything else ran through the pipeline.
    '''

    def fit(self, X, y=None):
        self.n_cols = X.shape[1]
        return self

    def transform(self, X, y=None):
        if X.shape[1] != self.n_cols:
            raise ValueError('Inconsistent columns')
        return X
