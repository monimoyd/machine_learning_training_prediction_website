# Machine Learning Training and Prediction Website

In this project I have developed a website where user can upload his datasets and then perform training on the dataset. Once
Training is done, prediction of input data will be done

# I. Problem Statement

Below are the requirements for the machine learning website

1. The website allows people to upload their own data:
        a. Images (10x2 - 80:20) upto (100*10 - 70-30)
        b. text samples (csv files) (min 100 - max 1000)*2-3 - 70:30)

2. Model will be trained on AWS EC2. AWS Lambda is used to preprocess, trigger EC2 and match user id with the files stored on EC2, and for inferencing (as discussed in the class, refer the video below)
3. Move the model to lambda and show inferencing
4. Transfer learning needs to be used
5. Limit max images per class to 100 and max classes to 10


## Usecases

The following diagram shows the main usecases for the website

![Usecase](/doc_images/Machine_Learning_Usecases.png)


# II. Architecture

The following diagram shows the architecture

![Project Report](/doc_images/architecture.png)


Architecture consists of following components
- UI consists of HTML, CSS, Javascript, JQuery
- AWS Labmda services REST APIs for the machine learning usecases
- EC2 instances are used for servicing train requests from AWS Lambda. Flask micro-framework alongwith uWSGI application server to launch the application 
and Nginx to act as a front end reverse proxy are installed on EC2 instance
- AWS DynamoDB is used as cloud database to manage persistence of various metadata and states of application
- AWS S3 is used for storing Datasets, Models


# III. Worflows

## i. Upload Dataset

User can upload datasets. Two types of datasets are 

1. Image Dataset: Image Datasets can be uploaded using zip file. The zip file consists of multiple  subdirectories where
each subdirectory correspding to a class label and images correspding to the class are stored. For example for animal 
dataset cat subdirectory consists of all the images belonging to cat class while dog subdirectory contains all the images
corresponding to Dog class

2. Sentiment Dataset: Sentiment dataset can be uploaded as a csv file. The csv file contains two columns, the first column
consists of any reviews/tweets/comments etc. while the second column contains sentiment score correpsonding to the 
reviews/tweets/comments

Workflow consists of following:

1. User uploads dataset from UI by providing following fields:

i. Dataset Name
ii. Dataset Description
iii. File corresponding to the dataset

2. Lambda APIs corresponding to upload dataset is invoked 
3. Dataset is validated to find all the constraints are satisfied
4. Upon successful validation of dataset, it is send to S3 bucket for storage
5. A record is added to Dataset table in AWS DynamoDB


ii. Train Dataset

Train workflow consists of following:

1. From UI, User selects a Dataset and clicks  trraining link to trigger training
2.  Lambda API corresponding to train function is invoked
3. A check is done whether server is busy or not. If busy, user is asked to try later
4. DB call on Dataset table is done to get metadata about the dataset
5. A check is made to see if the EC2 instance is currently stopped or running. If stoppped, a call is made to start the
instance
6. A post request is done to the Rest API for training exposed by EC2 flsk application by proving the dataset name
7. Dataset is downloaded from S3 bucket
8. Training is done on the EC2 instance
9. After successful training model is uploaded to S3 bucket
10. Dataset database in DynamoDB is updated with metadata information like name of Model file, bucket name for the Model
file, training accuracy, Validation accuracy



iii. Predict dataset

1. From UI, User selects a Dataset and Uploads a file to predict
2.  Lambda APIs corresponding to predict function is invoked
3. The metadata for model path ( and vocabulary for sentiment analysis) for the dataset is retried Dataset table in DynamoDB 
3. Model (and vocabulary in case of sentiment analyis) are downloaded to locally
4. Prediction is done using the model 
5. Prediction result is shown on UI


 
# IV. Database Design

Two tables in DynamoDB are created for tracking the application states

1. MachineStatus

It has following fields

i. name : instance id of EC2 instance
ii. State: Current state of EC2 instance (Busy/Free)


2. Dataset

It has following fields

i. name: Name of dataset
ii. description: Description of the dataset
iii. dataset_path: Bucket name and file name of the dataset in S3 
iv. model_path: Bucket name and file name of the dataset in S3 
v. vocab_path: Bucket name and file name of the dataset in S3 
vi. training_status: Status of last training done on the dataset (Training Successful/ Training Failed/ Training In Progress/Training Not Started)
vii. training_accuracy: Training accuracy of last successful training
viii. validation_accuracy: Validation accuracy of last successufl training


# V. Rest APIs

AWS Lambda is serviced by API Gateway service.

Each of the operations are identified by a unique operation id

sentiment_analysis_file_upload : Used for file upload for sentiment analysis
sentiment_analysis_train : Used for  train operation for sentiment analysis
sentiment_analysis_predict : Used for predict operation for sentiment analysis

image_classification_file_upload : Used for file upload for Image Classification
image_classification_train : Used for  train operation for Image Classification
image_classification_predict : Used for predict operation for Image Classification



The following REST APIs are used by Flask application in EC2

i. /train_sentiment_analysis: This API is used for performing training sentiment analysis. It takes dataset name 
path as input

ii. /predict_sentiment_analysis: his API is used for performing predicting sentiment analysis. It takes dataset name and 
text to predict as input

iii. /train_image_classification: This API is used for performing training image classification. It takes dataset name 
path as input


iii. /predict_image_classification: This API is used for performing training image classification. It takes dataset name 
path as input

# VI. Machine Learning Models used

For image classification, I have used the follwing
- Transfer learning by using pretrained Mobilenet. 
- Learning Rate: 0.01 momentum: 0.9 with SGD optimizer
- StepLR is used with step size: 6 and gamma: 0.1
- Epochs: 10


For sentiment Analysis, I have used following:

- LSTM Model
- Adam as optimizer
- Learning Rate: 2e-4
- Epochs: 10


# VII. Limitation and Future Enhancements

There are a few limitations for the application:

1. AWS lambda allows only 4.5 MB file upload, So user is not allowed to upload data file beyond this limitations
2. There is no authentication set, everyone is allowed


Future enhancements:

1. Add a authentication module
2. Allow user to edit datasets
3. Notification to user when training is completed



# VIII. Conclusion

In this project I have implemented a website from scratch for uploading, training, predicting Machine Learning Datasets. I 
have used various AWS services (EC2, Lamda, S3, DynamoDB). It was a very good learning opportunity


















