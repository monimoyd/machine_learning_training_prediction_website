# This file is your Lambda function
try:
    import unzip_requirements
except ImportError:
    pass

import base64
import json
from requests_toolbelt.multipart import decoder

import boto3
import time
import requests
import sys, traceback
import multiprocessing
import numpy as np


AWS_BUCKET_NAME = <AWS BUCKET>
AWS_ACCESS_KEY=<AWS ACCESS KEY>
AWS_SECRET_KEY=<AWS_SECRET_KEY>

def save_to_bucket(event, context):
    #s3 = boto3.resource('s3')
    #bucket = s3.Bucket(AWS_BUCKET_NAME)
    #path = 'testing_a_file.txt'
    #save_path = os.path.join(path, filename)
    content_type_header = event['headers']['content-type']
    #print("content_type_header:", content_type_header)
    #print("content loaded")
    body = base64.b64decode(event['body'])
    print('BODY LOADED')
    #print("body:", body)

    print('Getting content from body')
#    picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
    #decoded_fields = decoder.MultipartDecoder(body, content_type_header).parts[0]
    decoded_fields = decoder.MultipartDecoder(body, content_type_header)
    operation = decoded_fields.parts[0].content.decode('ascii')
    print("Operation:", operation)
    if operation == "sentiment_analysis_file_upload":
        dataset_name = decoded_fields.parts[1].content.decode('ascii')
        print("Dataset Name:", dataset_name)
        dataset_type = "sa"
        description = decoded_fields.parts[2].content.decode('ascii')
        print("Description:", description)
        file_content = decoded_fields.parts[3].content.decode('utf-8')
        #print("picture:", file_content)
        filename_base = dataset_name + "_" + dataset_type + "_" + str(time.time())
        dataset_filename = filename_base + ".csv"
        dataset_local_path = "/tmp/" + filename_base + ".csv"
    
        with open(dataset_local_path, "w") as f:
            f.write(file_content) 

        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
        )
        print("Saving dataset: " + dataset_name +  "to S3")
        s3 = session.resource('s3')
        s3.meta.client.upload_file(Filename=dataset_local_path, Bucket=AWS_BUCKET_NAME, Key=dataset_filename)

        print("Saved dataset: " + dataset_name +  "to S3")
        print("Saving dataset: " + dataset_name +  "to dynamoDB")
        dynamodb = None
        if not dynamodb:
            dynamodb = boto3.resource('dynamodb', region_name='ap-south-1', aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key= AWS_SECRET_KEY)

        table = dynamodb.Table('Dataset')
        response = table.put_item(
            Item={
            'name': dataset_name,
            'type': dataset_type,
	    'info': {
                'dataset_path': dataset_filename,
                'description': description,
                'traing_status': 'Not Started'
             }

        } )
        print("Response after saving to DB",response)

        if "200" in response:
            body = {
                "uploaded": "true",
                "bucket": AWS_BUCKET_NAME,
                "dataset_path": dataset_filename,
            }
        else:
            body = {
                "uploaded": "false",
                "bucket": AWS_BUCKET_NAME,
                "dataset_path": dataset_filename,
            }

        return {
            "statusCode": 200,
            "body": json.dumps(body)
        }
    elif operation == "image_classification_file_upload":
        handle_image_classification_file_upload(decoded_fields) 
    elif operation == "sentiment_analysis_train":
        handle_sentiment_analysis_train(decoded_fields) 
    elif operation == "image_classification_train":
        handle_image_classification_train(decoded_fields) 

def handle_image_classification_file_upload(decoded_fields):
    dataset_name = decoded_fields.parts[1].content.decode('ascii')
    print("Dataset Name:", dataset_name)
    dataset_type = "ec"
    description = decoded_fields.parts[2].content.decode('ascii')
    print("Description:", description)
    #file_content = decoded_fields.parts[3].content.decode('utf-8')
    file_content = decoded_fields.parts[3].content
    file_content_binary = np.frombuffer(file_content, dtype=np.uint8)
    #print("picture:", file_content)
    filename_base = dataset_name + "_" + dataset_type + "_" + str(time.time())
    dataset_filename = filename_base + ".zip"
    dataset_local_path = "/tmp/" + filename_base + ".zip"
    
    with open(dataset_local_path, "wb") as f:
        #f.write(str(file_content)) 
        f.write(file_content_binary) 

    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )
    print("Saving dataset: " + dataset_name +  "to S3")
    s3 = session.resource('s3')
    s3.meta.client.upload_file(Filename=dataset_local_path, Bucket=AWS_BUCKET_NAME, Key=dataset_filename)

    print("Saved dataset: " + dataset_name +  "to S3")
    print("Saving dataset: " + dataset_name +  "to dynamoDB")
    dynamodb = None
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb', region_name='ap-south-1', aws_access_key_id=AWS_BUCKET_NAME,
                aws_secret_access_key= AWS_SECRET_KEY)

    table = dynamodb.Table('Dataset')
    response = table.put_item(
        Item={
            'name': dataset_name,
            'type': dataset_type,
            'info': {
                'dataset_path': dataset_filename,
                'description': description,
                'training_status': 'Not Started'
              }

        } )
    print("Response after saving to DB",response)

    if "200" in response:
        body = {
            "uploaded": "true",
            "bucket": AWS_BUCKET_NAME,
            "dataset_path": dataset_filename,
        }
    else:
        body = {
            "uploaded": "false",
            "bucket": AWS_BUCKET_NAME,
            "dataset_path": dataset_filename,
        }

    return {
        "statusCode": 200,
        "body": json.dumps(body)
    }

def handle_sentiment_analysis_train(decoded_fields):
    print("Inside handle_sentiment_analysis_train")
    dataset_name = decoded_fields.parts[1].content.decode('ascii')
    print("Dataset Name:", dataset_name)
    machine_status = get_machine_status()
    print("machine_status:", machine_status)
    #if machine_status['info']['status'] == 'Busy':
    #    body = {
    #        "status": "System is busy other training requests. Please try after sometime",
    #    }
    #    return {
    #        "statusCode": 500,
    #        "body": json.dumps(body)
    #    }
    #else:
    resp = set_machine_status('Busy')
    p1 = multiprocessing.Process(target=process_train_sentiment_analysis, args=(dataset_name, ))
    p1.start()
    body = {
        "status": "Training is started. To check status please refresh training status page",
    }
    return {
        "statusCode": 200,
        "body": json.dumps(body)
    }

def handle_image_classification_train(decoded_fields):
    print("Inside handle_image_classification_train")
    dataset_name = decoded_fields.parts[1].content.decode('ascii')
    print("Dataset Name:", dataset_name)
    machine_status = get_machine_status()
    print("machine_status:", machine_status)
    #if machine_status['info']['status'] == 'Busy':
    #    body = {
    #        "status": "System is busy other training requests. Please try after sometime",
    #    }
    #    return {
    #        "statusCode": 500,
    #        "body": json.dumps(body)
    #    }
    #else:
    resp = set_machine_status('Busy')
    p1 = multiprocessing.Process(target=process_train_image_classification, args=(dataset_name, ))
    p1.start()
    body = {
        "status": "Training is started. To check status please refresh training status page",
    }
    return {
        "statusCode": 200,
        "body": json.dumps(body)
    }

def process_train_sentiment_analysis(dataset_name):
    try:
        print("process_train_sentiment_analysis")
        start_time = time.time()
        machine_running = is_machine_running()
        print("process_train_sentiment_analysis: machine_running: ", machine_running)
        if not machine_running:
            print("process_train_sentiment_analysis: machine not running: starting ec2 instance ")
            start_ec2_instance()
            print("ec2 instance started: waiting for 150 seconds")
            time.sleep(150)

        # TODO
        #url = 'http://ec2-15-206-161-171.ap-south-1.compute.amazonaws.com/train_sentiment_analysis'
        url = 'http://ec2-15-206-161-171.ap-south-1.compute.amazonaws.com:5000/train_sentiment_analysis'
        myobj = {'dataset_name': dataset_name}
        try:
            resp = requests.post(url, data = myobj)
            print("process_train_sentiment_analysis: Post response from ec2: ", resp)
            if "Internal Error" in resp:
                print("_train_sentimenet_analysis: Internal Error got from requests.post")
                return
        except Exception:
            print("process_train_sentimenet_analysis:Exception Post request failed")
            traceback.print_exc()
            #resp = set_machine_status('Free')
            return
        time.sleep(30)
        cur_time = time.time()
        while cur_time - start_time < 60*10:
            dataset = get_dataset(dataset_name, dataset_type)
            training_status = dataset['info']['training_status']
            print("process_train_sentimenet_analysis: Training Status: ", training_status)
            if training_status == 'Training Successful' or training_status =='Training Failed':
                print("process_train_sentimenet_analysis: training is successful or failed: exiting ")
                return
            time.sleep(15)
        print("process_train_sentiment_analysis: 30 minutes expired: exiting")
            
    finally:
        print("Finally stopping ec2 instance")
        #stop_ec2_instance()
        print("Finally making status free for ec2 instance")
        resp = set_machine_status('Free')
        print(resp)

def process_train_image_classificaion(dataset_name):
    try:
        print("process_train_image_classificaion")
        start_time = time.time()
        machine_running = is_machine_running()
        print("process_train_image_classification: machine_running: ", machine_running)
        if not machine_running:
            print("process_train_sentiment_analysis: machine not running: starting ec2 instance ")
            start_ec2_instance()
            print("ec2 instance started: waiting for 150 seconds")
            time.sleep(150)

        # TODO
        #url = 'http://ec2-15-206-161-171.ap-south-1.compute.amazonaws.com/train_sentiment_analysis'
        url = 'http://ec2-15-206-161-171.ap-south-1.compute.amazonaws.com:5000/train_image_classification'
        myobj = {'dataset_name': dataset_name}
        try:
            resp = requests.post(url, data = myobj)
            print("process_train_sentiment_analysis: Post response from ec2: ", resp)
            if "Internal Error" in resp:
                print("_train_sentimenet_analysis: Internal Error got from requests.post")
                return
        except Exception:
            print("process_train_sentimenet_analysis:Exception Post request failed")
            traceback.print_exc()
            #resp = set_machine_status('Free')
            return
        time.sleep(30)
        cur_time = time.time()
        while cur_time - start_time < 60*15:
            dataset = get_dataset(dataset_name, dataset_type)
            training_status = dataset['info']['training_status']
            print("process_train_sentimenet_analysis: Training Status: ", training_status)
            if training_status == 'Training Successful' or training_status =='Training Failed':
                print("process_train_image_classification: training is successful or failed: exiting ")
                return
            time.sleep(15)
        print("process_train_image_classification: 15 minutes expired: exiting")
            
    finally:
        print("Finally stopping ec2 instance")
        #stop_ec2_instance()
        print("Finally making status free for ec2 instance")
        resp = set_machine_status('Free')
        print(resp)



def is_machine_running() :
    instance_id = 'i-0c5c131090d1f9d95'
    ec2 = boto3.resource('ec2', region_name='ap-south-1',  aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key= AWS_SECRET_KEY)

    instances = ec2.instances.filter(
        Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
	
    running_instances_ids =[]
    for instance in instances:
        print(instance.id, instance.instance_type)
        running_instances_ids.append(instance.id)
	
    status_up = instance_id in running_instances_ids
    return status_up


def start_ec2_instance():
    print("start_ec2_instance")
    region = 'ap-south-1'
    instances = ['i-0c5c131090d1f9d95']
    ec2 = boto3.resource('ec2', region_name='ap-south-1',  aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key= AWS_SECRET_KEY)
    #ec2 = boto3.client('ec2', region_name=region)
    ec2.start_instances(InstanceIds=instances)
    print ('started your instances: ' + str(instances))
    return True

def stop_ec2_instance():
    print("stop_ec2_instance")
    region = 'ap-south-1'
    instances = ['i-0c5c131090d1f9d95']
    #ec2 = boto3.client('ec2', region_name=region)
    ec2 = boto3.resource('ec2', region_name='ap-south-1',  aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key= AWS_SECRET_KEY)
    ec2.stop_instances(InstanceIds=instances)
    
    print( 'stoped your instances: ' + str(instances))
    return True

def get_machine_status(dynamodb=None):
    machine_name = <MACHINE NAME>

    if not dynamodb:
        dynamodb = boto3.resource('dynamodb',  region_name='ap-south-1', aws_access_key_id=AWS_ACCESS_KEY,
         aws_secret_access_key= AWS_SECRET_KEY)

    table = dynamodb.Table('MachineStatus')

    try:
        response = table.get_item(Key={'machine_name': machine_name})
    except ClientError as e:
        print(e.response['Error']['Message'])
        return None
    else:
        return response['Item']

def set_machine_status(status, dynamodb=None):
    machine_name = <MACHINE NAME>

    if not dynamodb:
        dynamodb = boto3.resource('dynamodb',  region_name='ap-south-1', aws_access_key_id=AWS_ACCESS_KEY,
         aws_secret_access_key= AWS_SECRET_KEY)

    table = dynamodb.Table('MachineStatus')
    response = table.put_item(
            Item={
                'machine_name': machine_name,
                'info': {
                    'status': status,
                    'timestamp': str(time.time())
                }
           })



def get_dataset(dataset_name, dataset_type, dynamodb=None):
    AWS_SECRET_KEY=AWS_SECRET_KEY
    BUCKET_NAME = AWS_BUCKET_NAME

    if not dynamodb:
        dynamodb = boto3.resource('dynamodb',  region_name='ap-south-1', aws_access_key_id=AWS_ACCESS_KEY,
         aws_secret_access_key= AWS_SECRET_KEY)

    table = dynamodb.Table('Dataset')

    try:
        response = table.get_item(Key={'name': dataset_name, 'type':dataset_type})
    except ClientError as e:
        print(e.response['Error']['Message'])
        return None
    else:
        return response['Item']

