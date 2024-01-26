import sys
import os

current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


import config
import pandas as pd
import logging

# ETL
#Extract
def extract (source:str)-> pd.DataFrame:
    """
    Extract data from source
    """
    try:
        logging.info(f"Extracting data from {source}")
        return pd.read_parquet(source)
    except Exception as e:
        logging.error(f"Error extracting data from {source}:{e}")
        return pd.DataFrame()

# Transform Step
def transform(data:pd.DataFrame) -> pd.DataFrame:
    """
    Transform the data
    """    
    try:
        logging.info(f"Transforming data")
        data['reviewText']=data['reviewText'].str.lower()
        data.drop(columns=['reviewerID', 'asin', 'reviewerName', 
                           'unixReviewTime', 'vote', 'style', 'image'], inplace=True)
        data['summary']=data['summary'].str.lower()
        data.dropna(inplace=True)
        ### drop unverified reviews
        data = data.loc[data.verified==True]
        data.drop(columns=['verified'], inplace=True)

        ### length of reviewText
        data['reviewLength'] = data.reviewText.apply(lambda text: len(text.split(' ')))

        # drop outliers
        q1=data.reviewLength.quantile(0.25)
        q3=data.reviewLength.quantile(0.75)
        IQR=q3-q1
        upper_bound=q3+(1.5*IQR)
        lower_bound=q1-(1.5*IQR)
        data=data[(data.reviewLength<upper_bound)& (data.reviewLength>lower_bound)]
        return data
    except Exception as e:
        logging.error(f"Error transforming data from:{e}")
        return pd.DataFrame()

# LOAD
def load(data: pd.DataFrame, target:str) -> None:
    """
    Load the data to target
    """
    try:
        logging.info(f"Loading data to {target}")
        data.to_parquet(target)
    except Exception as e:
        logging.error(f"Error loading the data to {target}:{e}")


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting feature pipeline")

    #Extract
    data=extract(config.DATA_SOURCE)

    # Transform
    data= transform(data)

    #Load
    load(data, config.TRAIN_DATA_SOURCE)
    print(data.head(10))
    logging.info("Featuring pipeline completed successfully")
    logging.info("Exiting feature pipeline")
    logging.shutdown
    exit()