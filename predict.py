import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets,transforms,models
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import json
import argparse
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Load a model from checkpoint and make prediction on an Image")
    parser.add_argument('--img_path', default='./flowers/test/28/image_05230.jpg',type=str,help='set the image path')
    parser.add_argument('--topk', default=5,type=int,help='set the no. of topk')
    parser.add_argument('--gpu_mode',default=False,type=bool,help='set the gpu mode')
    parser.add_argument('--ckpt_pth',default='ckpt.pth',type=str,help='set the ckpt path')
    args = parser.parse_args()
    return args

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    saved_model = checkpoint['model']
    saved_optimizer = checkpoint['optimizer']
    saved_epoch = checkpoint['epoch']
    return saved_model,saved_optimizer,saved_epoch

def process_image(image):
    image = image.resize((256,256))
    image = image.crop((16,16,240,240))
    image = np.array(image)
    image = image/255
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    image = (image - mean)/std
    image = np.transpose(image,(2,0,1))
    return image.astype(np.float32)

def predict(image_path,model,device,topk=5):
    img = Image.open(image_path)
    img = process_image(img)
    img = np.expand_dims(img,0)
    img = torch.from_numpy(img)
    model.eval()
    inputs = img.to(device)
    logits = model.forward(inputs)
    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)
    return (e.data.numpy().squeeze().tolist() for e in topk)

def main():
    args = parse_args()
    img_path = args.img_path
    gpu_mode = args.gpu_mode
    topk = args.topk
    ckpt_pth = args.ckpt_pth
    
    print('='*10+'Params'+'='*10)
    print('Image path:       {}'.format(img_path))
    print('Load model from:  {}'.format(ckpt_pth))
    print('GPU mode:         {}'.format(gpu_mode))
    print('TopK:             {}'.format(topk))
    
    model,__,__ = load_checkpoint(ckpt_pth)
    class_names = model.class_names
    # set GPU mode
    if gpu_mode and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device  = torch.device("cpu")
    print('Current device: {}'.format(device))
    model.to(device)
    
    # Label Mapping 
    with open('cat_to_name.json','r') as f:
        cat_to_name = json.load(f)
        
    #Predict
    print('='*10+'Predict'+'='*10)
    probs,classes = predict(img_path,model,device,topk)
    flower_names = [cat_to_name[class_names[e]] for e in classes]
    for prob, flower_name in zip(probs,flower_names):
        print('{:20}: {:.4f}'.format(flower_name,prob))
        
if __name__ == '__main__':
    main()
    
    
    