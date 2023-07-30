# TENREC: Recommendation System Dataset and Its Use in Cold Start
The project is inspired and based on the paper [Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems](https://arxiv.org/abs/2210.10629)

## Description

The user or visitor cold start simply means that a recommendation engine meets a new visitor for the first time. because there is no user history about them, the system does not know the personal preferences of the user. Getting to know your visitors is crucial in creating a great user experience for them.

In this project, we discuss the use of Tenrec dataset for the cold start tast in recommendation systems. This is done via training the dataset on both *BERT4Rec* and *Peterrec* in order to find the best model suitable for the dataset, in addition to the performance the dataset itself. Furthermore, the project aims for generalization. Therefore, we use Amazon Ratings (Beauty Products) dataset in order to check its results on both models. 


## Dataset
The datasets used are:
* Tenrec Data [Download](https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html):
The Tenrec dataset suite is a collection of user behavior logs from Tencent's recommendation platforms, QQ BOW (QB) and QQ KAN (QK). The dataset includes user feedback from four different scenarios: QK-video, QK-article, QB-video, and QB-article. In the QK/QB platforms, items can be either news articles or videos.
    1. cold_data.csv
    2. cold_data_1.csv
    3. cold_data_0.7.csv
    4. cold_data_0.3.csv
    5. sbr_data_1M.csv
* ratings_Beauty.csv [Download](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings): 
The Amazon - Ratings (Beauty Products) dataset is a collection of user-generated ratings and reviews related to beauty products available on Amazon's e-commerce platform. It contains information about the interactions between users and various beauty items, providing valuable data for building and evaluating recommendation systems and sentiment analysis models.


In order to use these files, they must be saved in a folder called *Tenrec* as follows:
```
Tenrec/<file name>
```

## Prerequisites
The environments that are needed to be installed are the following:

* Python
* Tensorflow
* Pytorch
* sklearn

To install these requirements:
```
pip install -r requirements.txt
```

## Benchmark
This repository includes 4 notebooks:
    1. `Tenrec_visualization.ipynb`: contains statistical information about the Tenrec's cold_start datasets, in addition to the sbr_data_1M dataset.
    2. `amazon_ratings.ipynb`: contains statistical information about the Amazon Beauty Product Reviews dataset.
    3. `main-Bert4Rec.ipynb`: contains the BERT4Rec models' training using the different pre-processing methods.
    4. `main_peter.ipynb`: contains the Peterrec models' training using the different pre-processing methods.

In addition to the use of 4 packages, which are:
    1. `bert4pytorch.py`: BERT4Rec model building
    2. `peter4pytorch.py`: Peterrec model building
    3. `preprocessing.py`: includes the 6 pre-processing techniques applied on the datasets
    4. `train.py`: for training the models.

#### The project is divided into two parts for the two models. 
* For BERT4Rec, run `main-Bert4Rec.ipynb`
* For Peterrec, run `main_peter.ipynb`
    
## Results
* The BERT4REC model has shown promising results for both the Tenrec and Amazon datasets.
* The results indicate that PETERREC has limitations in effectively handling the cold start problem, especially for the Tenrec dataset. 
* As we can see our pipeline are suitable to be used on any dataset available for cold start problem.
* From all the results we can noticed that the best preprocessing trial is Thired & Sixth trial which give the best results.
* The Third & Sixth trials are simmilar the only difference is the dataset we are using.
* The best pre processing method is when we use one file to extract both source and target, and the target item is the last ineracted item by user.


## License
This dataset is licensed under a CC BY-NC 4.0 International 

You can find the license [here](https://creativecommons.org/licenses/by-nc/4.0/).




