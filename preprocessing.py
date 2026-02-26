"""
Data preprocessing module with .fit and .transform methods.

- Categorical: one-hot encode if < 10 levels; remove feature if >= 10 levels.
- Missing: impute features (median or mean); remove rows with missing target in fit.
- Correlation: remove features so no pairwise correlation > threshold (default 0.99).
- Standardize: zero mean, unit variance.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    Preprocessor with separate fit and transform.
    Accepts DataFrame or ndarray; if ndarray, categorical_columns should be provided.
    """

    def __init__(
        self,
        categorical_columns: Optional[List[Union[str, int]]] = None,
        max_categorical_levels: int = 10,
        impute_strategy: str = "median",
        correlation_threshold: float = 0.99,
    ):
        """
        Args:
            categorical_columns: Column names or indices to treat as categorical.
                If None and X is a DataFrame, inferred from dtypes (object, category).
            max_categorical_levels: Categoricals with more than this many unique
                values are dropped; others are one-hot encoded.
            impute_strategy: 'median' or 'mean' for imputing missing feature values.
            correlation_threshold: Remove features so no pairwise correlation
                exceeds this value.
        """
        if impute_strategy not in ("median", "mean"):
            raise ValueError("impute_strategy must be 'median' or 'mean'")
        self.categorical_columns = categorical_columns
        self.max_categorical_levels = max_categorical_levels
        self.impute_strategy = impute_strategy
        self.correlation_threshold = correlation_threshold

        # Fitted state
        self._feature_names_in: Optional[List[str]] = None
        self._numeric_cols: Optional[List[str]] = None
        self._cat_cols_onehot: Optional[List[str]] = None
        self._cat_cols_drop: Optional[List[str]] = None
        self._imputer: Optional[SimpleImputer] = None
        self._onehot_encoder: Optional[OneHotEncoder] = None
        self._onehot_col_names: Optional[List[str]] = None
        self._correlation_drop_indices: Optional[np.ndarray] = None
        self._scaler: Optional[StandardScaler] = None
        self._fitted: bool = False

    def _ensure_dataframe(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)

    def _get_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        if self.categorical_columns is not None:
            return [str(c) for c in self.categorical_columns if str(c) in X.columns]
        return [
            c for c in X.columns
            if X[c].dtype in ("object", "category") or X[c].dtype.name == "object"
        ]

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "DataPreprocessor":
        """
        Fit the preprocessor. Drops rows where y is missing (if y is provided).
        """
        X = self._ensure_dataframe(X)
        X.columns = [str(c) for c in X.columns]

        self._fit_row_indices_kept = np.arange(len(X))
        if y is not None:
            y = pd.Series(y) if not isinstance(y, pd.Series) else y
            mask = np.asarray(~pd.isna(y)).ravel()
            if not mask.all():
                X = X.iloc[mask].reset_index(drop=True)
                self._fit_row_indices_kept = np.where(mask)[0]

        self._feature_names_in = list(X.columns)
        cat_cols = self._get_categorical_columns(X)
        numeric_cols = [c for c in X.columns if c not in cat_cols]

        cat_cols_onehot = []
        cat_cols_drop = []
        for c in cat_cols:
            n_unique = X[c].nunique()
            if n_unique >= self.max_categorical_levels:
                cat_cols_drop.append(c)
            else:
                cat_cols_onehot.append(c)

        self._numeric_cols = numeric_cols
        self._cat_cols_onehot = cat_cols_onehot
        self._cat_cols_drop = cat_cols_drop

        # Impute numeric columns
        if numeric_cols:
            self._imputer = SimpleImputer(strategy=self.impute_strategy)
            self._imputer.fit(X[numeric_cols])
        else:
            self._imputer = None

        # One-hot encode low-level categoricals
        if cat_cols_onehot:
            self._onehot_encoder = OneHotEncoder(
                handle_unknown="ignore",
                drop=None,
                sparse_output=False,
            )
            self._onehot_encoder.fit(X[cat_cols_onehot].astype(str))
            self._onehot_col_names = [
                f"{col}_{val}"
                for col, vals in zip(
                    cat_cols_onehot,
                    self._onehot_encoder.categories_,
                )
                for val in vals
            ]
        else:
            self._onehot_encoder = None
            self._onehot_col_names = []

        # Build combined matrix: [imputed_numeric | one-hot]
        if numeric_cols:
            X_num = self._imputer.transform(X[numeric_cols])
        else:
            X_num = np.empty((len(X), 0))
        if cat_cols_onehot:
            X_cat = self._onehot_encoder.transform(X[cat_cols_onehot].astype(str))
        else:
            X_cat = np.empty((len(X), 0))
        A = np.hstack([X_num, X_cat])

        # Remove highly correlated features (greedy)
        drop_idx = self._find_correlation_drops(A)
        self._correlation_drop_indices = drop_idx
        keep_mask = np.ones(A.shape[1], dtype=bool)
        keep_mask[drop_idx] = False
        A_reduced = A[:, keep_mask]

        self._scaler = StandardScaler()
        self._scaler.fit(A_reduced)
        self._fitted = True
        return self

    def _find_correlation_drops(self, A: np.ndarray) -> np.ndarray:
        """Return indices of columns to drop so no pairwise correlation > threshold."""
        if A.shape[1] <= 1:
            return np.array([], dtype=int)
        corr = np.corrcoef(A.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0)
        drop_set = set()
        while True:
            i, j = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
            if np.abs(corr[i, j]) <= self.correlation_threshold:
                break
            to_drop = j if j not in drop_set else i
            drop_set.add(to_drop)
            corr[to_drop, :] = 0
            corr[:, to_drop] = 0
        return np.array(sorted(drop_set), dtype=int)

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit first.")
        X = self._ensure_dataframe(X)
        X.columns = [str(c) for c in X.columns]
        if list(X.columns) != self._feature_names_in:
            if len(X.columns) == len(self._feature_names_in) and np.all(
                [str(a) == str(b) for a, b in zip(X.columns, self._feature_names_in)]
            ):
                pass
            else:
                raise ValueError(
                    "transform X has different columns than fit. "
                    f"Fit: {self._feature_names_in}, got: {list(X.columns)}"
                )

        if self._numeric_cols and self._imputer is not None:
            X_num = self._imputer.transform(X[self._numeric_cols])
        else:
            X_num = np.empty((len(X), 0))
        if self._cat_cols_onehot:
            X_cat = self._onehot_encoder.transform(
                X[self._cat_cols_onehot].astype(str)
            )
        else:
            X_cat = np.empty((len(X), 0))
        A = np.hstack([X_num, X_cat])

        keep_mask = np.ones(A.shape[1], dtype=bool)
        keep_mask[self._correlation_drop_indices] = False
        A_reduced = A[:, keep_mask]
        return self._scaler.transform(A_reduced)

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> np.ndarray:
        """Fit and transform in one step. Returns transformed data for rows kept in fit (no missing y)."""
        self.fit(X, y)
        X_kept = self._ensure_dataframe(X).iloc[self._fit_row_indices_kept]
        return self.transform(X_kept)
