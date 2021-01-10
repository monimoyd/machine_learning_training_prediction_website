import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import CrossEntropyLoss



def train(model, device, train_loader, optimizer, epoch, train_losses,train_acc, l1_param=0.0 ):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  local_train_losses = []
  local_train_acc = []
  for batch_idx, train_data in enumerate(pbar):
    # get samples  
    data, target = train_data['X'].to(device), train_data['Y'].to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    #loss = F.nll_loss(y_pred, target)
    criterion = CrossEntropyLoss()
    loss = criterion(y_pred, target)
    regularization_loss = 0.0

    for param in model.parameters():
      if param.dim() > 1 :
          regularization_loss += param.norm(1)
    
    regularization_loss *= l1_param
    loss += regularization_loss


    local_train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    local_train_acc.append(100*correct/processed)
  train_acc.append(sum(local_train_acc) / len(local_train_acc))
  train_losses.append(sum(local_train_losses) / len(local_train_losses))

def test(model, device, test_loader, test_losses, test_acc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for test_data in test_loader:
            data, target = test_data['X'].to(device), test_data['Y'].to(device)
            output = model(data)
            criterion = CrossEntropyLoss()
            test_loss += criterion(output, target)
            #loss = criterion(y_pred, target)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
