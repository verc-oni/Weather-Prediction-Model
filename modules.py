from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import datetime


def format_datetime(dt:str) -> datetime.datetime:
    date = datetime.datetime.strptime(dt[:23], '%Y-%m-%d %H:%M:%S.%f')
    return date


# Weather Data Cleaning Model
class Weather:
    def __init__(self, file, ext=None) -> None:
        self.file:str = file
        self.ext = ext
       
    def _ext(self, string):
        if hasattr(pd, f"read_{string}"):
            return eval(f'pd.read_{string}')
        else:
            raise ValueError('Pandas as NO attribute called read_{string}'.format(string))
    
   
    def _extension(self):
        if self.file.count('.') == 1:
            if not self.ext:
                string = self.file[self.file.index('.')+1:]
                self.ext = string
                return self._ext(string.lower())
        else:
            raise ValueError('File Object Url is Invalid')
        
    def retreive_data(self):
        wd:pd.DataFrame = self._extension()(self.file)
        data = wd.copy(deep=True)
        return data
    
    def process_data(self):
        data = self.retreive_data()
        data_date = data['Formatted Date']
        data = data.drop(['Formatted Date'], axis=1)
        copy_date_data, copy_date_index = [], []
        for i, date in enumerate(data_date):
            copy_date_index.append(i)
            copy_date_data.append(format_datetime(date).strftime('%Y-%m-%d'))
        formatted_date = pd.Series(data=copy_date_data, index=copy_date_index)
        data = data.assign(Date=formatted_date)
        month = pd.DatetimeIndex(data['Date']).month
        year = pd.DatetimeIndex(data['Date']).year
        day = pd.DatetimeIndex(data['Date']).day
        data = data.assign(Year = year, Month = month, Day = day)
        return data

    
    def return_data(self):
        data = self.process_data()
        return data
    

# Class For The Prediction Model
class Predictor:
    def __init__(self, data:pd.DataFrame, features: list | None=None, target: str='Month', model:object=None, **kwargs):
        self.data = data
        self.target = target
        self.features = features
        self.model = Ridge
        self.p_model = PolynomialFeatures
        self.predictors = 'Temperature (C)'
        if model:
            self.model = model
        if kwargs:
            self.p_model = kwargs.pop('pipeline_model', None)
    
    def set_predictors(self):
        if not self.predictors:
            columns = self.data.columns
            print('Selected Prediction Values against the Time Value...\n')
            for index, i in enumerate(columns):
                print(f"{index+1}). {i}")
            try:
                value = int(input(">>"))
            except ValueError as e:
                print(f"Invalid Input, expected Int, got {type(value).__name__}")
            else:
                self.predictors = columns[value-1]
    
    def prepare_variables(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data[self.target].to_numpy().reshape(-1, 1), self.data[self.predictors], test_size=0.2, random_state=0)
    
    def pipeline_model(self):
        pipe = make_pipeline(self.p_model(), self.reg_model())
        return pipe
    
    def reg_model(self):
        model = self.model(alpha=.1) if isinstance(self.model, Ridge) else self.model()
        return model
    
    def initialize(self):
        self.pipeline = self.pipeline_model()
        self.reg = self.reg_model()

    def trainset_xy(self):
        self.prepare_variables()
        return [self.x_train, self.y_train]
    
    def train(self, trainset):
        self.reg.fit(*trainset)
        self.pipeline.fit(*trainset)
    
    def predict(self):
        self.reg_pred = self.reg.predict(self.x_test)
        self.pipe_pred = self.pipeline.predict(self.x_test)
    
    def error_prox(self):
        self.reg_mse = mean_squared_error(self.y_test, self.reg_pred)
        self.pipe_mse = mean_squared_error(self.y_test, self.pipe_pred)
    
    def representation(self):
        print(f"Ridge Mean Error : {self.reg_mse} ({self.reg_mse/np.mean(self.reg_pred)*100:3.3}%)")
        print(f"Pipeline Mean Error : {self.pipe_mse} ({self.pipe_mse/np.mean(self.pipe_pred)*100:3.3}%)")
        print(f"Ridge Score : {self.reg.score(self.x_train, self.y_train)}")
        print(f"Pipeline Score : {self.pipeline.score(self.x_train, self.y_train)}")
  
         
    def start(self):
        eval_list = ['set_predictors','initialize', ['train', 'trainset_xy'], 'predict', 'error_prox', 'representation']
        for i in eval_list:
            if isinstance(i, list):
                eval(f'self.{i[0]}(self.{i[1]}())')
                continue
            eval(f'self.{i}()')    
    
    
    