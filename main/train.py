## import libraries for training
import sys
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
from utils import *
from time import time
import pyarrow.parquet as pq
import multiprocessing as mp
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_lr_finder import LRFinder
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_workers = mp.cpu_count()
## Writing the loss and results
if not os.path.exists("./logs/"): 
    os.mkdir("./logs/")
log = Logger()
log.open("logs/%s_log_trainTesting.txt")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |----- Train -----|----- Valid----|----Train----|-----------|\n')
log.write('mode     iter     epoch    |       loss      |        mAP    |   Accuracy  |   time    |\n')
log.write('-------------------------------------------------------------------------------------------\n')

#
train_acc_history = []
val_acc_history = []
train_loss_history = []
## Training the model
def train(train_loader,model,criterion,optimizer,epoch,valid_accuracy,start,starter):
    losses = AverageMeter()
    model.train()
    model.training=True
    correct_train = 0
    total_train = 0
    running_loss = 0.0
    for i,(images,target,fnames) in enumerate(train_loader):
        img = images.to(device, non_blocking=True)
        label = target.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            logits = model(img).to(device)
        loss = criterion(logits, label)
        losses.update(loss.item(),images.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()        
        optimizer.zero_grad()
        scheduler.step()

        running_loss += loss.item()

        _, predicted = torch.max(logits.data, 1)
        total_train += label.size(0)
        correct_train += (predicted == label).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        ending = time()
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f      |      %0.4f      |      %0.3f     | %0.3f | %s | %s ' % (\
                "train", i, epoch,losses.avg,valid_accuracy[0],train_accuracy,time_to_str((timer() - start),'min'),"Finish with:{} ".format(round(ending - starter,2)))
        print(message , end='',flush=True)
    log.write("\n")
    log.write(message)
    
    train_acc_history.append(train_accuracy)
    train_loss_history.append(train_loss)

    return [train_loss_history,train_acc_history,losses.avg]

# Validating the model
def evaluate(val_loader,model,criterion,epoch,train_loss,start):
    model.to(device)
    model.eval()
    model.training=False
    map = AverageMeter()
    with torch.no_grad():
        for i, (images,target,fnames) in enumerate(val_loader):
            img = images.to(device,non_blocking=True)
            label = target.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)
            
            

            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5,img.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    | %s' % (\
                    "val", i, epoch, train_loss, map.avg,time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
        log.write("\n")  
        log.write(message)
    return [map.avg]

## Computing the mean average precision, accuracy 
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5

def plot_history(train_loss_history, train_acc_history, val_acc_history,epochs):
    epochs = list(range(1, epochs + 1))
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(epochs,train_loss_history, label='Loss',marker='o')
    ax1.set_title('Loss over epochs on Resnet50')
    ax1.set_ylabel('Loss')
    ax1.axis([0, 20, 0, 5])
    ax1.text(3, 9, 'batch_size:32 | lr = 0.002 | AdamW')

    ax1.legend(loc='upper right')

    ax2.plot(val_acc_history, label='mAP',marker='o')
    ax2.plot(train_acc_history, label='Accuracy',marker='o')
    ax2.set_title('Train Accuracy over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')

    plt.show()
######################## load file and get splits #############################

#train_imlist = pd.read_csv("main/test.csv")
#train_imlist = pd.read_parquet("main/train.parquet", engine='fastparquet')
train_gen = knifeDataset("main/train.parquet",mode="train")
################ speed up training test num worker ##########
train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=num_workers,prefetch_factor=2)
#val_imlist = pd.read_parquet("main/val.parquet", engine='fastparquet')
val_gen = knifeDataset("main/val.parquet",mode="val")
val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=num_workers,prefetch_factor=2)

## Loading the model to run
model = timm.create_model('resnet18', pretrained=False,num_classes=config.n_classes)
model.to(device)

############################# Parameters #################################
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs * len(train_loader), eta_min=0,last_epoch=-1)
criterion = nn.CrossEntropyLoss().to(device)

############################# Training #################################
start_epoch = 0
val_metrics = [0]
scaler = torch.cuda.amp.GradScaler()
start = timer()
starter = time()
#train

def lr(model, optimizer, criterion):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
    lr_finder.plot()
    return lr_finder.get_best_lr()

for epoch in range(0,config.epochs):
    lr = get_learning_rate(optimizer)
    train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,start, starter)
    val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics[2],start)
    val_acc_history.append(val_metrics[0].item())
plot_history(train_metrics[0],train_metrics[1],val_acc_history,30)


