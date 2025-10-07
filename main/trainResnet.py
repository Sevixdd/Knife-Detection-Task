## import libraries for training
import sys
import warnings
from backbone import ResNet
from model import FBSD
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
from model import FBSD
import pyarrow.parquet as pq
import multiprocessing as mp
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_workers = mp.cpu_count()
## Writing the loss and results
if not os.path.exists("./logs/"): 
    os.mkdir("./logs/")
log = Logger()
log.open("logs/%s_log_trainTesting.txt")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |----- Train -----|----- Valid-----|----Train----|----------|\n')
log.write('mode     iter     epoch    |       loss      |        mAP     |   Accuracy  |   time   |\n')
log.write('-------------------------------------------------------------------------------------------\n')

## Training the model
def train(train_loader,model,criterion,optimizer,epoch,valid_accuracy,start,starter):
    losses = AverageMeter()
    model.train()
    model.training=True
    train_loss = 0
    correct = 0
    total = 0
    for i,(images,target,fnames) in enumerate(train_loader):
        idx = i
        img = images.to(device, non_blocking=True)
        label = target.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            output_1, output_2, output_3, output_concat = model(img)
        # Adjust the loss calculation based on your specific criteria
        loss1 = criterion(output_1, label) * 2
        loss2 = criterion(output_2, label) * 2
        loss3 = criterion(output_3, label) * 2
        concat_loss = criterion(output_concat, label)
        loss = (loss1 + loss2 + loss3 + concat_loss)/4   
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 
        optimizer.zero_grad()
        optimizer.step()
        scheduler.step()

        _, predicted = torch.max((output_1+ output_2+ output_3 + output_concat).data, 1)
        total += label.size(0)
        correct += predicted.eq(label.data).cpu().sum()

        train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item() )

        train_acc = 100. * float(correct) / total

        train_loss = train_loss / (idx + 1)
        ending = time()
        #valid_accuracy = torch.tensor(valid_accuracy)
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f      |      %0.5f      |     %0.3f     |     %0.3f%%  (%d/%d) | %s | %s ' % (\
                "train", i, epoch,train_loss / (idx + 1), valid_accuracy[0] if epoch==0 else valid_accuracy.item(),train_acc, correct, total,time_to_str((timer() - start),'min'),"Finish with:{} second, ".format(round(ending - starter,2)))
        print(message , end='',flush=True)
    log.write("\n")
    log.write(message)



    return [train_loss]


def test(net, testloader):
    net.eval()
    correct_com = 0
    total = 0

    softmax = nn.Softmax(dim=-1)

    for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                output_1, output_2, output_3, output_concat = net(inputs)
                outputs_com = output_1 + output_2 + output_3 + output_concat

            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct_com += predicted_com.eq(targets.data).cpu().sum()
    test_acc_com = 100. * float(correct_com) / total

    return test_acc_com 
# Validating the model
def evaluate(val_loader, model,criterion,epoch,train_loss,start):
    model.cuda()
    model.eval()
    model.training = False
    map = AverageMeter()
    correct_com = 0
    total = 0
    with torch.no_grad():
        for i, (images, target, fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)
            
            with torch.no_grad():
                output_1, output_2, output_3, output_concat = model(img)
                outputs_com = output_1 + output_2 + output_3 + output_concat
                # You need to use the appropriate output for your task
                # For example, you might use fm4_boost, but adjust this based on your requirements
                _, max_values = torch.max(outputs_com, dim=1, keepdim=True)
                outputs_shifted = outputs_com - max_values
                preds = torch.softmax(outputs_shifted, dim=1)
            #predicted_com = torch.amax(outputs_com, dim=(1, 2))
            #preds = predicted_com.softmax(1)
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5, img.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    | %s' % (\
                    "val", i, epoch, train_loss[0], map.avg,time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
        log.write("\n")  
        log.write(message)
    return map.avg
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


######################## load file and get splits #############################

#train_imlist = pd.read_csv("main/test.csv")
#train_imlist = pd.read_parquet("main/train.parquet", engine='fastparquet')
train_gen = knifeDataset("main/train.parquet",mode="train")
################ speed up training test num worker ##########
train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=num_workers,prefetch_factor=2)
#val_imlist = pd.read_parquet("main/val.parquet", engine='fastparquet')
val_gen = knifeDataset("main/test.parquet",mode="val")
val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=num_workers,prefetch_factor=2)

## Loading the model to run

model = FBSD(class_num=config.n_classes, arch='resnet50')
#model = timm.create_model('tf_efficientnet_b0', pretrained=True,num_classes=config.n_classes)
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
for epoch in range(0,config.epochs):
    lr = get_learning_rate(optimizer)
    train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,start, starter)
    val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,start)
    ## Saving the model
filename = "Resnet-50AdamW" + str(epoch + 1)+ str(config.batch_size) + str(config.learning_rate) + str(config.new_weight_decay)+".pt"
torch.save(model.state_dict(), filename)
    

   
