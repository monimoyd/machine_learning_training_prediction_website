import torch
from torch.utils.data import DataLoader

def get_test_accuracy(model, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)      
            predicted = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
            total += labels.size(0)
    
    return 100.0* correct / total
	
def get_accuracy_per_class(model, testloader, device,  num_classes=10):
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)      
            predicted = outputs.argmax(dim=1, keepdim=True)
            c =  predicted.eq(labels.view_as(predicted)).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    return class_correct, class_total