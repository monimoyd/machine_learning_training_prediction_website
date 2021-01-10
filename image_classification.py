import sys

sys.path.append('./models/')  #Path for model
sys.path.append('./utils/') #Path for Graph Plottinng, Training & Test/Evaluation Logic
sys.path.append('./data_loaders/') #Path for DataLoad
sys.path.append('./data_transformations/') #Path for DataTransformation

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader

from data_transformations.mobilenet_data_transformation import get_train_transform, get_test_transform
from data_loaders.mobilenet_data_loader import ImageDataset
from utils.train_test_utils import train, test

import os
import torch
from torchsummary import summary
from models.mobilenet_model import get_model
from utils.train_test_utils import train, test
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import zipfile
from PIL import image

def get_classes(path):
    return [f for f in os.listdir(path)]

def get_size_of_dataset(path): 
    count = 0
    sub_dir_list = [f for f in os.listdir(path)]
    for sub_dir in sub_dir_list:
        path_sub_dir = os.path.join(path, sub_dir)
        count += len([f for f in os.listdir(path_sub_dir) if os.path.isfile(os.path.join(path_sub_dir, f))])
    return count




def train_val_dataset(path, dataset_size, val_split=0.2):
    #size_of_dataset = get_size_of_dataset(path)
    train_idx, val_idx = train_test_split(list(range(dataset_size)), test_size=val_split)
    #print("train_idx: ", train_idx)
    #print("val_idx: ", val_idx)
    datasets = {}
    #datasets['train'] = Subset(dataset, train_idx)
    #datasets['val'] = Subset(dataset, val_idx)
    train_transform = get_train_transform()
    test_transform = get_test_transform()
    datasets['train'] = ImageDataset('./images', train_idx, train_transform )
    datasets['val'] = ImageDataset('./images', val_idx, test_transform )
    return datasets
	
import numpy as np
from torch.utils.data import Dataset,DataLoader
#test_transform = get_test_transform()
def train_image_classification(dataset_name, dataset_path):

    BATCH_SIZE=64
    classes= get_classes('./images')
    output_size = len(classes)
    dataset_size = get_size_of_dataset('./images')

    temp_images_folder = 'images/' + str(time.time())
    if os.path.isdir(temp_images_folder):
        os.rmdir(temp_images_folder)
    os.mkdir(temp_images_folder)
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(temp_images_folder)



#dataset = ImageFolder('./images', transform=Compose([Resize((224,224)),ToTensor()]))

#dataset = TrainImageDataset('./images', transform=train_transform)
#print(len(dataset))
    datasets = train_val_dataset('./images', dataset_size, val_split=0.2)
    print(len(datasets['train']))
    print(len(datasets['val']))



#sample_test = next(iter(test_dl))
#print(sample_test['X'].shape, sample_test['Y'].shape)

    train_dl = DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
#sample_train = next(iter(train_dl))
#print(sample_train['X'].shape, sample_train['Y'].shape)

    test_dl = DataLoader(datasets['val'], batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = get_model( output_size, True)

    model = model.to(device)
    summary(model, input_size=(3, 224, 224))



    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    model_name = dataset_name + "_ic.pth"
    MODEL_PATH = './model_files/' + model_name


    optimizer = optim.SGD(model.classifier[1].parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
    best_test_accuracy = 0.0
    EPOCHS = 2
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train(model, device, train_dl, optimizer, epoch, train_losses,train_acc )
        test(model, device, test_dl, test_losses, test_acc)
        t_acc = test_acc[-1]
        if t_acc > best_test_accuracy:
            best_test_accuracy = t_acc
            best_train_accuracy = train_acc[-1]
            torch.save(model.state_dict(), MODEL_PATH)
        #model.to('cpu')
        #traced_model= torch.jit.trace(model, torch.rand(1,3,244,244))
        #traced_model.save(PATH)
        #compiled_model= torch.jit.script(model)
        #torch.jit.save(compiled_model,PATH)
        scheduler.step()
	
    dataiter = iter(test_dl)
    data = dataiter.next()

    with torch.no_grad():
        images, labels = data['X'].to(device), data['Y'].to(device)
        outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(15)))

    correct = 0
    with torch.no_grad():
        for test_data in test_dl:
            data, target = test_data['X'].to(device), test_data['Y'].to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    print("Test Accuracy: ", 100. * correct / len(test_dl.dataset))							  
    result_dict  ={}
    result_dict['training_status'] = 'Training Successful'
    result_dict['training_accuracy'] = str(best_train_acc)
    result_dict['validation_accuracy'] = str(best_valid_acc)
    result_dict['model_path'] = model_name
    result_dict['num_output_nodes'] = str(output_size)
    result_dict['prediction_classes'] = str(classses)

    if os.path.isdir(temp_images_folder):
        os.rmdir(temp_images_folder)

    return result_dict

def transform_image(image_bytes):
    try:
        #transformations = transforms.Compose([
        #    transforms.Resize(255),
        #    transforms.CenterCrop(224),
        #    transforms.ToTensor(),
        test_transform = get_test_transform()
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)

def get_prediction(model, image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()

def get_list_from_class_str(class_list_str):
    class_list_str_skip = class_list_str[2: len(class_list_str) -3]
    class_list = class_list_str_skip.split("', '")
    return class_list

def predict_image_classification(dataset_name,predict_image_path, model_path, num_output_nodes, prediction_classes ) :
    try:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = get_model( num_output_nodes)
        model.load_state_dict(torch.load(model_path));
        model = model.to(device)
        model.eval()

        file = io.open(predict_image_path, "rb", buffering = 0)
        predict_bytes_stream = file.read()
        prediction_class_list = get_list_from_class_str(prediction_classes)
       
        prediction_id = get_prediction(image_bytes=predict_bytes_stream)
        prediction = prediction_class_list[ prediction_id ]
        resp = {}
        resp['prediction_value'] =  prediction

    except Exception as e:
        print(repr(e))
        resp = {}
        resp['prediction_value']=  "Failed to predict becuase of internal error"
    return resp

#train_image_classification()
