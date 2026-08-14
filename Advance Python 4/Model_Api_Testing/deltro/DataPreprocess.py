import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression


class DataPreprocessing:

    def __init__(self, file, target):

        
        self.df = pd.read_csv(file)
        self.target = target

        self.process()

    def process(self):

        self.df = self.df.drop_duplicates()

        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]

        X = X.fillna(0)

        for col in X.select_dtypes("object"):
            X[col] = LabelEncoder().fit_transform(
                X[col].astype(str)
            )
            
        for col in X.columns:
            if X[col].nunique() <= 1:
                X = X.drop(col, axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        if y.dtype == "object" or y.nunique() <= 2:
            self.model = LogisticRegression()
        else:
            self.model = LinearRegression()


        self.model.fit(self.X_train,self.y_train)

        return self.model
    

ml = DataPreprocessing(
    "Social_Network_Ads.csv",
    "EstimatedSalary"
)
