from flask import Flask, jsonify, request, redirect, render_template
import os, pickle
import numpy as np

import random
import math
import time
import image_classification
import sentiment_analysis
import boto3
from botocore.exceptions import ClientError
from flask import request
import multiprocessing
import sys, traceback
import os.path
from os import path

app = Flask(__name__)
app.secret_key = "secret key"
from flask import jsonify, make_response

AWS_ACCCESS_KEY='AKIAQMLZ23B6KCIYGGFC'
AWS_SECRET_KEY='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'
BUCKET_NAME = 'image-uploads-2222'





# URL Routes
@app.route('/')
def index():
    print("calling index()")
    sentiment_analysis.perform_training('tweets', 'tweets1.csv')
    
    return render_template('result_new.html')

@app.route('/train_sentiment_analysis', methods=['POST'])
def train_sentiment_analysis():
    print("calling train_sentiment_analysis()")
    resp = {}
    dataset_name = request.form.get('dataset_name')
    print("train_sentiment_analysis: dataset_name: ", dataset_name)
    if dataset_name:
         dataset_obj= get_dataset(dataset_name, "sa")
         print("train_sentiment_analysis: dataset_obj: ", dataset_obj)
         if dataset_obj :
             print("train_sentiment_analysis: before calling process ofr senttiment_analysis")
             p1 = multiprocessing.Process(target=perform_train_sentiment_analysis, args=(dataset_obj, ))
             p1.start()
             resp["training_status"] = "Training Started"
             print("train_sentiment_analysis: Training started")
             save_dataset(dataset_obj, resp)
             return make_response(jsonify(resp), 200)
         else:
             print("train_sentiment_analysis: Internal Error: Training could not be started as dataset_obj is None")
             resp["training_status"] = "Internal Error: Training could not be started"
             return make_response(jsonify(resp), 500)
    else:
        print("train_sentiment_analysis: Internal Error: Training could not be started as dataset_name is None")
        resp["training_status"] = "Internal Error: Training could not be started"
        return make_response(jsonify(resp), 500)

@app.route('/predict_sentiment_analysis', methods=['POST'])
def predict_sentiment_analysis():
    print("calling predict_sentiment_analysis()")
    resp = {}
    dataset_name = request.form.get('dataset_name')
    input_text = request.form.get('input_text')
    print("predict_sentiment_analysis: dataset_name: ", dataset_name)
    if dataset_name:
         dataset_obj= get_dataset(dataset_name, "sa")
         print("predict_sentiment_analysis: dataset_obj: ", dataset_obj)
         if dataset_obj :
             print("predict_sentiment_analysis: before calling process ofr senttiment_analysis")
             #p1 = multiprocessing.Process(target=perform_train_sentiment_analysis, args=(dataset_obj, ))
             #p1.start()
             model_name = dataset_obj['info']['model_path']
             print("predict_sentiment_analysis: model_name: ", model_name)
             model_path = "model_files/" + model_name
             if not path.exists(model_path):
                 print("Downloaded model file locally")
                 download_s3_file(model_name, model_path): 
             else:
                 print("Model file already exists")

             vocab_name = dataset_obj['info']['vocab_path']
             print("predict_sentiment_analysis: vocab_name: ", vocab_name)
             vocab_path = "vocab_files/" + vocab_name
             if not path.exists(vocab_path):
                 download_s3_file(vocab_name, vocab_path): 
                 print("Downloaded Vocab file locally")
             else:
                 print("Vocab file already exists")

             size_of_vocab = int(dataset_obj['info']['size_of_vocab'])
             print("predict_sentiment_analysis: size_of_vocab: ", size_of_vocab)
             num_output_nodes = int(dataset_obj['info']['num_output_nodes'])
             print("predict_sentiment_analysis: num_output_nodes: ", num_output_nodes)
            
             error = False
             try:
                 print("predict_sentiment_analysis: Calling sentiment_analysis.predict_sentiment_analysis")
                 resp = sentiment_analysis.predict_sentiment_analysis(dataset_name,input_text, model_path, vocab_path, size_of_vocab, num_output_nodes )
             except Exception:
                 error = True
                 print("predict_sentiment_analysis: Error while calling sentiment_analysis.predict_sentiment_analysis")
                 resp["prediction_status"] = "Internal Error: Predction could not be started"
                 traceback.print_exc()

             if error == False:
                 return make_response(jsonify(resp), 200)
             else:
                 return make_response(jsonify(resp), 500)
         else:
             print("train_sentiment_analysis: Internal Error: Training could not be started as dataset_obj is None")
             resp["prediction_status"] = "Internal Error: Predction could not be started"
             return make_response(jsonify(resp), 500)
    else:
        print("train_sentiment_analysis: Internal Error: Training could not be started as dataset_name is None")
        resp["training_status"] = "Internal Error: Training could not be started"
        return make_response(jsonify(resp), 500)


@app.route('/train_image_classification', methods=['POST'])
def train_image_classfication():
    resp = {}
    dataset_name = request.form.get('dataset_name')
    print("train_image_classification: dataset_name: ", dataset_name)
    if dataset_name:
         dataset_obj= get_dataset(dataset_name, "ec")
         print("train_image_classification: dataset_obj: ", dataset_obj)
         if dataset_obj :
             print("train_image_classification: before calling process ofr senttiment_analysis")
             p1 = multiprocessing.Process(target=perform_train_image_classification, args=(dataset_obj, ))
             p1.start()
             resp["training_status"] = "Training Started"
             print("train_sentiment_analysis: Training started")
             save_dataset(dataset_obj, resp)
             return make_response(jsonify(resp), 200)
         else:
             print("train_sentiment_analysis: Internal Error: Training could not be started as dataset_obj is None")
             resp["training_status"] = "Internal Error: Training could not be started"
             return make_response(jsonify(resp), 500)
    else:
        print("train_sentiment_analysis: Internal Error: Training could not be started as dataset_name is None")
        resp["training_status"] = "Internal Error: Training could not be started"
        return make_response(jsonify(resp), 500)

@app.route('/predict_image_classification', methods=['POST'])
def predict_image_classification():
    print("Inside image_classification")
    resp = {}
    dataset_name = request.form.get('dataset_name')
    prediction_file_name = request.form.get('prediction_file_name')
    print("predict_image_classification: dataset_name: ", dataset_name)
    if dataset_name and prediction_file_name:
         dataset_obj= get_dataset(dataset_name, "ec")
         print("predict_image_classification: dataset_obj: ", dataset_obj)
         if dataset_obj :
             print("predict_image_classification: before calling process ofr senttiment_analysis")
             #p1 = multiprocessing.Process(target=perform_train_image_classification, args=(dataset_obj, ))
             #p1.start()
             prediction_file_path = "prediction_files/" + prediction_file_name
             download_s3_file(prediction_file_name, prediction_file_path): 
             model_path = "model_files/" + model_name
             if not path.exists(model_path):
                 print("Downloaded model file locally")
                 download_s3_file(model_name, model_path): 
             else:
                 print("Model file already exists")
             num_output_nodes = dataset_obj['info']['num_output_nodes']
             prediction_classes = dataset_obj['info']['prediction_classes']
             error = False
             try:
                 print("predict_image_classification: Calling sentiment_analysis.predict_sentiment_analysis")
                 resp = image_classification.predict_image_classification(dataset_name,predict_image_path, model_path, num_output_nodes, prediction_classes )
             except Exception:
                 error = True
                 print("predict_sentiment_analysis: Error while calling sentiment_analysis.predict_sentiment_analysis")
                 resp["prediction_status"] = "Internal Error: Predction could not be started"
                 traceback.print_exc()
             if not error:
                 return make_response(jsonify(resp), 200)
             else:
                 return make_response(jsonify(resp), 500)
         else:
             print("predict_image_classification: Internal Error: prediction could not be started as dataset_obj is None")
             resp["prediction_status"] = "Internal Error: Prediction could not be started"
             return make_response(jsonify(resp), 500)
    else:
        print("predict_image_classification: Internal Error: prediction could not be started as dataset_name is None")
        resp["prediction_status"] = "Internal Error: Prediction could not be started"
        return make_response(jsonify(resp), 500)


def get_dataset(dataset_name, dataset_type, dynamodb=None):
    AWS_ACCCESS_KEY='AKIAQMLZ23B6KCIYGGFC'
    AWS_SECRET_KEY='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'
    BUCKET_NAME = 'image-uploads-2222'

    if not dynamodb:
        dynamodb = boto3.resource('dynamodb',  region_name='ap-south-1', aws_access_key_id='AKIAQMLZ23B6KCIYGGFC', 
         aws_secret_access_key= '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')

    table = dynamodb.Table('Dataset')

    try:
        response = table.get_item(Key={'name': dataset_name, 'type':dataset_type})
    except ClientError as e:
        print(e.response['Error']['Message'])
        return None
    else:
        return response['Item']

def perform_train_sentiment_analysis(dataset_obj):
    dataset_name = dataset_obj['name']
    dataset_path = dataset_obj['info']['dataset_path']
    print("perform_train_sentiment_analysis:  dataset_name: ", dataset_name,"dataset_path: ", dataset_path)
    local_downloaded_path = "text_files/" + dataset_path
    download_s3_file(dataset_path, local_downloaded_path)
    print("perform_train_sentiment_analysis: download_s3_file for local_downloaded_path done")
    try:
        result_dict = sentiment_analysis.perform_training(dataset_name, local_downloaded_path)
    except Exception:
        print("perform_train_sentiment_analysis:Training Failed")
        result_dict = {}
        result_dict['training_status'] = 'Training Failed'
        #ex_type, ex, tb = sys.exc_info()
        #traceback.print_tb(tb)
        traceback.print_exc()
    print("perform_train_sentiment_analysis: sentiment_analysis.train_model done")
    save_dataset(dataset_obj, result_dict)
    print("perform_train_sentiment_analysis: save_dataset done")

def perform_train_image_classification(dataset_obj):
    dataset_name = dataset_obj['name']
    dataset_path = dataset_obj['info']['dataset_path']
    print("perform_train_image_classification:  dataset_name: ", dataset_name,"dataset_path: ", dataset_path)
    local_downloaded_path = "image_files/" + dataset_path
    download_s3_file(dataset_path, local_downloaded_path)
    print("perform_train_image_classification: download_s3_file for local_downloaded_path done")
    try:
        print("perform_train_image_classification: calling train_image_classificatin()")
        result_dict = image_classification.train_image_classification(dataset_name, dataset_path)
    
        print("perform_train_image_classification: calling train_image_classification")

        result_dict = sentiment_analysis.perform_training(dataset_name, local_downloaded_path)
    except Exception:
        print("perform_train_sentiment_analysis:Training Failed")
        result_dict = {}
        result_dict['training_status'] = 'Training Failed'
        #ex_type, ex, tb = sys.exc_info()
        #traceback.print_tb(tb)
        traceback.print_exc()
    print("perform_train_image_classification: sentiment_analysis.train_model done")
    save_dataset(dataset_obj, result_dict)
    print("perform_train_image_classification: save_dataset done")

def download_s3_file(dataset_path, local_path): 
    AWS_ACCCESS_KEY = 'AKIAQMLZ23B6KCIYGGFC'
    AWS_SECRET_KEY = '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'
    BUCKET_NAME = 'image-uploads-2222'
    s3 = boto3.resource('s3', aws_access_key_id = 'AKIAQMLZ23B6KCIYGGFC', aws_secret_access_key = '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')
    #s3 = boto3.client('s3')
	
    bucket = s3.Bucket('image-uploads-2222')
    bucket.download_file( dataset_path, local_path )


def save_dataset(dataset_obj, result_dict):
    AWS_ACCCESS_KEY='AKIAQMLZ23B6KCIYGGFC'
    AWS_SECRET_KEY='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'
    BUCKET_NAME = 'image-uploads-2222'
    dynamodb = boto3.resource('dynamodb', region_name='ap-south-1', aws_access_key_id='AKIAQMLZ23B6KCIYGGFC', aws_secret_access_key='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')

    table = dynamodb.Table('Dataset')
    if result_dict['training_status'] == 'Training Successful':
        model_path = result_dict['model_path']
        vocab_path = result_dict['vocab_path']
        upload_s3("model_files/" + model_path, BUCKET_NAME,model_path )
        upload_s3("vocab_files/" + vocab_path, BUCKET_NAME,vocab_path )
        if dataset_obj['type'] == 'sa':
            response = table.put_item(
                Item={
                    'name': dataset_obj['name'],
                    'type': dataset_obj['type'],
	            'info': {
                        'dataset_path': dataset_obj['info']['dataset_path'],
                        'description': dataset_obj['info']['description'],
                        'training_status': result_dict['training_status'],
                        'model_path': model_path,
                        'vocab_path': vocab_path,
                        'training_accuracy': result_dict['training_accuracy'],
                        'validation_accuracy': result_dict['validation_accuracy'],
                        'num_output_nodes': result_dict['num_output_nodes'],
                        'size_of_vocab': result_dict['size_of_vocab'],
                    }
               }) 
        elif dataset_obj['type'] == 'ec':
            response = table.put_item(
                Item={
                    'name': dataset_obj['name'],
                    'type': dataset_obj['type'],
	            'info': {
                        'dataset_path': dataset_obj['info']['dataset_path'],
                        'description': dataset_obj['info']['description'],
                        'training_status': result_dict['training_status'],
                        'model_path': model_path,
                        'training_accuracy': result_dict['training_accuracy'],
                        'validation_accuracy': result_dict['validation_accuracy'],
                        'num_output_nodes': result_dict['num_output_nodes'],
                        'prediction_classes': result_dict['prediction_classes'],

                    }
               }) 
    else:
        response = table.put_item(
            Item={
                'name': dataset_obj['name'],
                'type': dataset_obj['type'],
	        'info': {
                    'dataset_path': dataset_obj['info']['dataset_path'],
                    'description': dataset_obj['info']['description'],
                    'training_status': result_dict['training_status'],
                }
           }) 

    if "200" in response:
        print("Updated DB successfully")


def upload_s3(local_file_path, bucket_name, s3_file):
    AWS_ACCCESS_KEY='AKIAQMLZ23B6KCIYGGFC'
    AWS_SECRET_KEY='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'
    BUCKET_NAME = 'image-uploads-2222'

    session = boto3.Session(
    aws_access_key_id='AKIAQMLZ23B6KCIYGGFC',
    aws_secret_access_key='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')
    s3 = session.resource('s3')
    s3.meta.client.upload_file(Filename=local_file_path, Bucket='image-uploads-2222', Key=s3_file)



#@app.route('/', methods=['POST'])
#def predict():
#    if request.method == 'POST':
#        tweet = request.form['tweet']
#        translation, attention = translate_sentence(tweet, SRC, TRG, model, device)
#        print(f'predicted trg = {translation}')
#        translation_str = " ".join([tr for tr in translation if tr != '<eos>'])
#    return render_template('result.html', prediction=translation_str)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
