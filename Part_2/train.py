import os
from torch.utils.data import Dataset, DataLoader
from dataset import get_loader
import torch
import torch.nn as nn
import torch.nn.functional as F          
from network import R2U_Net
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt


####______________Loading Dataset____________###
train_loader, train_dataset = get_loader('train')
valid_loader, valid_dataset = get_loader('val')


####______________Hyperparameters____________###
num_classes=20
epochs=250
lr = 1e-6
b1 = 0.5
b2 = 0.999
decay_ratio = random.random()*0.8
decay_epoch = int(epochs*decay_ratio)


####______________Model Instance____________###
device = torch.device(f'cuda:{7}' if torch.cuda.is_available() else "cpu")
model=R2U_Net(img_ch=3,output_ch=num_classes)
model.to(device)
optimizer = torch.optim.Adam(list(model.parameters()),lr, [b1, b2])
criterion = nn.NLLLoss2d()

model = torch.nn.DataParallel(model, device_ids = [7, 1, 3])

####___________ERROR HANDLING______________###
#model = torch.nn.DataParallel(model)
##
print(device)


####______________Training____________###
train_loss_values = []
val_loss_values = []
PATH = "./model-R2U-Net.cpt"
print("Starting training process..")

for e in range(epochs):
    # Training
    train_loss = 0
    model = model.train()
    i = 0
    for data in train_loader:
        im = Variable(data[0].to(device))
        label = Variable(data[1].to(device))
        out = model(im)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, label)
        if i%50 == 0:
           print(i,len(train_loader), loss)
        i = i + 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data

    # Validation
    model = model.eval()
    eval_loss = 0
    for data in valid_loader:
        with torch.no_grad():
          im = Variable(data[0].to(device))
          label = Variable(data[1].to(device))
          out = model(im)
          out = F.log_softmax(out, dim=1)
          loss = criterion(out, label)
          eval_loss += loss.data
              
    # Saving cpt
    print("Saving checkpoint for epoch", e)
    torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH+str(e))
    os.system("curl --upload-file ./model-R2U-Net.cpt" + str(e) + " http://transfer.sh/model-FCN-Net.cpt" + str(e))
    print("\n")
    os.system("rm ./model-R2U-Net.cpt" + str(e))
    print("deleted")
    train_loss_values.append( train_loss / len(train_loader))
    val_loss_values.append(eval_loss / len(valid_loader))

    epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Valid Loss: {:.5f}'.format(
        e, train_loss / len(train_loader), eval_loss / len(valid_loader)))
   
    print(epoch_str)

train_loss_values = torch.FloatTensor(train_loss_values)
val_loss_values = torch.FloatTensor(val_loss_values)
plt.plot(train_loss_values.detach().cpu().numpy())
plt.plot(val_loss_values.detach().cpu().numpy())
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["Train_loss", "Val_loss"])
plt.savefig('./loss.png')
