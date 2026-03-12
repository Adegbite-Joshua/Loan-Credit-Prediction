import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, FunctionTransformer, PowerTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector



class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        ## Feature engineering      

        # Extract years and months
        X['years'] = X['Credit_History_Age'].str.extract('(\d+)\s*Years?').astype(float)
        X['months'] = X['Credit_History_Age'].str.extract('(\d+)\s*Months?').astype(float)
        
        X['age_years'] = X['years'] + X['months'] / 12
        X['total_months'] = X['years'] * 12 + X['months']
        
        X["Annual_Income_Per_Age"] = X["Annual_Income"] / X["age_years"]
        X["Has_Debt"] = (X["Outstanding_Debt"] > 0).astype(int)
        X["Debt_to_Income_Ratio"] = X["Outstanding_Debt"] / X["Annual_Income"]
        X["Loan_Per_Bank"] = X["Num_Bank_Accounts"] / (X["Num_of_Loan"] + 1)
        X["Annual_Income_Per_Loan"] = X["Annual_Income"] / (X["Num_of_Loan"] + 1)
        X["Outstanding_Debt_Per_Monthly_Balance"] = X["Outstanding_Debt"] / (X["Monthly_Balance"] + 1)
        
        return X


def iqr_cap(X):
    X = X.copy()
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.clip(X, lower_bound, upper_bound)


class NumericalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, skew_threshold=0.5):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):      
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns        
        skewness = X[self.numeric_cols_].skew()

        self.high_skew_cols_ = skewness[abs(skewness) > self.skew_threshold].index.tolist()
        self.low_skew_cols_ = skewness[abs(skewness) <= self.skew_threshold].index.tolist()

        transformers = []

        if self.high_skew_cols_:
            transformers.append(
                ('high_skew',
                Pipeline([
                    ('cap', FunctionTransformer(iqr_cap, feature_names_out="one-to-one")),
                    ('power', PowerTransformer(method='yeo-johnson')),
                    ('scale', RobustScaler())
                ]),
                self.high_skew_cols_)
            )

        if self.low_skew_cols_:
            transformers.append(
                ('low_skew',
                Pipeline([
                    ('cap', FunctionTransformer(iqr_cap, feature_names_out="one-to-one")),
                    ('scale', RobustScaler())
                ]),
                self.low_skew_cols_)
            )
            
        self.column_transformer_ = ColumnTransformer(
            transformers,
            remainder='passthrough'
        )

        self.column_transformer_.fit(X)

        # Save output column names
        self.feature_names_ = self.column_transformer_.get_feature_names_out()

        return self

    def transform(self, X):
        X = pd.DataFrame(X)

        transformed = self.column_transformer_.transform(X)

        return pd.DataFrame(
            transformed,
            columns=self.feature_names_,
            index=X.index
        )
        


