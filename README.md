## About Dataset

Research Hypothesis: This study hypothesizes that there are significant associations between the diagnostic characteristics of patients, including age, menopause status, tumor size, presence of invasive nodes, affected breast, metastasis status, breast quadrant, history of breast conditions, and their breast cancer diagnosis result. Data Collection and Description:The dataset of 213 patient observations was obtained from the University of Calabar Teaching Hospital cancer registry over 24 months (January 2019–August 2021). The data includes eleven features: year of diagnosis, age, menopause status, tumor size in cm, number of invasive nodes, breast (left or right) affected, metastasis (yes or no), quadrant of the breast affected, history of breast disease, and diagnosis result (benign or malignant).Notable Findings:Upon preliminary examination, the data shows variations in diagnosis results across different patient features. A noticeable trend is the higher prevalence of malignant results among patients with larger tumor sizes and the presence of invasive nodes. Additionally, postmenopausal women seem to have a higher rate of malignant diagnoses.Interpretation and Usage:The data can be analyzed using statistical and machine learning techniques to determine the strength and significance of associations between patient characteristics and breast cancer diagnosis. This can contribute to predictive modeling for the early detection and diagnosis of breast cancer.However, the interpretation must consider potential limitations, such as missing data or bias in data collection. Furthermore, the data reflects patients from a single hospital, limiting the generalizability of the findings to wider populations.The data could be valuable for healthcare professionals, researchers, or policymakers interested in understanding breast cancer diagnosis factors and improving healthcare strategies for breast cancer. It could also be used in patient education about risk factors associated with breast cancer.

About Dataset

    S/N = Unique identification for each patient.
    Year=The year diagnosis was conducted.
    Age = Age of patient at the time of diagnose.
    Menopause = Whether the patient is pro or postmenopausal at the time diagnose,0 MEANS THAT THE PATIENT HAS REACHED MENOPAUSE WHILE 1 MEANS THAT THE PATIENT HAS NOT REACHED MENOPAUSE YET.
    Tumor size = The size in centimeter of the excised tumor.
    Involved nodes = The number of axillary lymph nodes that contain metastatic,"CODED AS A BINARY DISTRI UTION OF EITHER PRESENT OR ASENT. 1 MEANS PRESENT, 0 MEANS ABSENT."
    Breast = If it occurs on the left or right side,"CODED AS A BINARY DISTRIBUTION 1 MEANS THE CANCER HAS SPREAD, 0 MEANS IT HASN'T SPREAD YET."
    Metastatic = If the cancer has spread to other part of the body or organ.
    Breast quadrant = The gland is divided into 4 sections with nipple as a central point.
    History = If the patient has any history or family history on cancer,"1 means there is a history of cancer , 0 means no history."
    Diagnosis result = Instances of the breast cancer dataset.

## The project is divided into three (3) parts/stages:

    - Load data for preprocesing and save preprocessed data for (EDA)
    - Perform Exploratory Data Analysis on the preprocessed data and save model
    - Load saved model and test the model

## Environment Configuration (Installing virtual Env):

    -pip install pipenv

    Using github;
        -create a repo with your github account
        -clone the repo on your local directory
        -change directory from your local repo to the cloned github repo

    Installing Packages
        pipenv install:
        -jupyter notebook
        -pandas
        -numpy
        -matplotlib
        -seaborn
        -scikit-learn
        -pyarrow

## Starting/Stopping Virtual Env

    Starting Notebook
        -pipenv shell
        -jupyter-notebook

    Stoping Notebook
        -Ctrl+c

    Deactiving Virtual Env
        -exit

## Load data reviewing of data:

    Import libraries
        -import numpy as np
        -import pandas as pd

    Load and perform overview of dataset
        -df.head()
        -df.tail().T
        -df.info()
        -df.shape()
        -df.dtypes
        -df.isnull().sum()
        -unique instances() etc

## Data preprocessing:

    Import libraries
        -import numpy as np
        -import pandas as pd

    Data Preprocessing
        -Normalize the column names to lower case
        -Drop the ID column
        -Remove the (+) sign on the Dependants column
        -Fill the NaN in the (Dependants, Credit_History, Loan_Amount, Gender, Self_Employed) columns
        -Replace categorical column(Loan-Status) with integers
        -Save cleaned dataset

## Exploratory Data Analysis (EDA):

    Import libraries
        -import numpy as np
        -import pandas as pd
        -import matplotlib.pyplot as plt
        -import seaborn as sns
        -from sklearn.model_selection import train_test_split
        -from sklearn.feature_extraction import DictVectorizer
        -from sklearn.linear_model import LogisticRegression
        -from sklearn.metrics import accuracy_score
        -import pickle

## Target Variable Analysis:

    -Load the cleaned loan dataset
    -Perform a target variable analysis
    -Build a Validation Framework
    -Divide the dataset into three (3)
        -Training dataset 60%
        -Validation dataset 20%
        -Testing dataset 20%

## Feature Engineering:

    -Seperate the dataset into numerical attributes and categorical attributes
    -perform the one-hot encoding
    -convert the dataframe into dict
    -DictVectorizer
    -(fit) the train_dict

## Train The Model:

    -LogisticRegression
    -compaire predicted truth vrs ground truth

The predictions of the model: a two-column matrix. The first column contains the probability that the target is zero (the application will be approved). The second column contains the opposite probability (the target is one, and the application will be rejected).
The output of the (probabilities) is often called soft predictions. These tell us the probability of rejection as a number between zero and one. It’s up to us to decide how to interpret this number and how to use it.

## Saving The Model:

    -import pickle
    -specifyging where to save the file
    -save the model

## Testing the Model:

    Load libraries
        -import numpy as np
        -import pandas as pd
        -import pickle

    -load the saved model
    -Load patient Data to predict status (Positive/Negative)

    Models's verdict:

        -if prediction >= 0.5, patient test positive
        -if prediction <= 0.5, patient test negative
