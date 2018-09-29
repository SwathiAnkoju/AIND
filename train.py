import torch
from torchvision import datasets, models,transforms
from torch import nn
from torch import optim
import os
import copy
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classifier')
    parser.add_argument('--data_root',default='flowers',type=str,help='set data dir')  
    parser.add_argument('--model',type=str,choices=["densenet","alexnet","resnet"],required=True,default='densenet121')
    parser.add_argument('--learning_rate',type=float,default=0.001)
    parser.add_argument('--hidden_layers',nargs='+',type=int, default=None, help='List of integers, the sizes of hidden layers')
    parser.add_argument('--epochs', type=int, default=5,help='No. of training epochs')
    parser.add_argument('--gpu_mode',type=bool, default=False,help='set the gpu modes')
    parser.add_argument('--ckpt_pth',default='ckpt.pth',type=str,help='set the save name')
    args = parser.parse_args()
    return args

def trainModel(data_loader,dataset_size,model,criterion,optimizer,device,epochs=10):
    
    best_acc = 0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1,epochs))
        print('-'*10)
        
        for phase in ['train','valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for images,labels in data_loader[phase]:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(images)
                    _, prediction = torch.max(output,1)
                    loss = criterion(output,labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item()*images.size(0)
                running_corrects += torch.sum(prediction == labels.data)
                
            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]
            
            print('{} Loss : {:.4f}, Accuracy : {:.4f}'.format(phase,epoch_loss,epoch_acc))
            
            if phase == 'valid' and epoch_acc > best_acc :
                best_acc = epoch_acc
        print()            
    print('Best Val Accuracy:{:4f}'.format(best_acc))
    
    return model

# Building Classifier 
def build_network(num_in_features, hidden_layers,num_out_features):
    
    classifier = nn.Sequential()
    if hidden_layers == None:
        classifier.add_module('drop',nn.Dropout(0.5))
        classifier.add_module('bn',nn.BatchNorm1d(num_in_features))
        classifier.add_module('fc0', nn.Linear(num_in_features, 102))
    else:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        classifier.add_module('drop0', nn.Dropout(0.5))
        classifier.add_module('bn',nn.BatchNorm1d(num_in_features))
        classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))    
        classifier.add_module('relu0', nn.ReLU(inplace=True))
        classifier.add_module('drop1', nn.Dropout(0.5))
        classifier.add_module('bn1',nn.BatchNorm1d(hidden_layers[0]))

        for i, (h1, h2) in enumerate(layer_sizes):           
            classifier.add_module('fc'+str(i+1), nn.Linear(h1, h2))
            classifier.add_module('relu'+str(i+1), nn.ReLU(inplace=True))
            classifier.add_module('drop'+str(i+1), nn.Dropout(0.5))
        classifier.add_module('output', nn.Linear(hidden_layers[-1], num_out_features))
 
    return classifier

def main():
    args = parse_args()
    data_dir = args.data_root
    gpu_mode = args.gpu_mode
    model_name = args.model
    lr = args.learning_rate
    hidden_layers = args.hidden_layers
    epochs = args.epochs
    ckpt_pth = args.ckpt_pth
    
    print('='*10+'Params'+'='*10)
    print('Data dir:      {}'.format(data_dir))
    print('Model:         {}'.format(model_name))
    print('Hidden Layers: {}'.format(hidden_layers))
    print('Learning rate: {}'.format(lr))
    print('Epochs:        {}'.format(epochs))
    
    #Defining Validations for training, testing and validation sets
    data_transforms = {
                'train':transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                       ]),

                'test':transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                  ]),

                'valid':transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                    ]),
    }
    #Load the datasets with Image Folder
    image_dataset = {x: datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])
                 for x in ['train','test','valid']}
    # Defining Dataloaders using image datasets and train transforms
    data_loader = {x:torch.utils.data.DataLoader(image_dataset[x],batch_size=32,shuffle=True)
              for x in ['train','test','valid']}
    dataset_size = {x: len(image_dataset[x]) for x in ['train','test','valid']}
    class_names = image_dataset['train'].classes
    if gpu_mode and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print('Current Device: {}'.format(device))
    
    #Choosing the pre-trained network
    if model_name == 'resnet':
        model = models.resnet34(pretrained=True)
        num_in_features = model.fc.in_features
    elif model_name == 'densenet':
        model = models.densenet121(pretrained=True)
        num_in_features = model.classifier.in_features
    else:
        print('Unknown Model, Please choose Resnet or densenet')
        
    # Freezing Parameters to avoid Back Propagation
    for param in model.parameters():
        param.requires_grad = False
    #Redesign Classifier
    classifier = build_network(num_in_features,hidden_layers,102)
    if model_name == 'resnet':
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    elif model_name == 'densenet':
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    else:
        pass
    
    print('='*10 + ' Architecture ' + '='*10)
    print('Classifier Architecture:')
    print(classifier)
    
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    #Train 
    print('='*10 + ' Train ' + '='*10)
    model = trainModel(data_loader,dataset_size,model,criterion,optimizer,device,epochs)
    #Test
    print('='*10 + ' Test ' + '='*10)
    model.eval()
    accuracy = 0
    for images,labels in data_loader['test']:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        # class with highest probability is our predicted class
        equality = (labels.data == output.max(1)[1])
        #accuracy is no.of correct predictions divided by all predictions. 
        accuracy += equality.type_as(torch.FloatTensor()).mean()
    print("Test Accuracy: {:.3f}".format(accuracy/len(data_loader['test'])))
    # Save the Checkpoint
    print('='*10 + ' Save ' + '='*10)
    model.class_to_idx = image_dataset['train'].class_to_idx
    model.class_names = class_names
    checkpoint = {'epoch' : epochs,
                  'model' : model,
                  'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint,ckpt_pth)
    print('Save the trained model to {} '.format(ckpt_pth))
   
if __name__ == '__main__':
    main()
    
    

    
    
    
    
    
    
    
                
                
                
            
                
        
        
    
    
 