import os
from torch.utils.data import Dataset, DataLoader
from dataset import get_loader
import torch
import torch.nn as nn
import torch.nn.functional as F          
from network import fcn
from torch.autograd import Variable
import matplotlib.pyplot as plt


####___________Loading Dataset______________####
num_classes=20
train_loader, train_dataset = get_loader('train')
valid_loader, valid_dataset = get_loader('val')

####___________Hyperparameteres______________####
epochs=250
lr = 1e-2
wd = 1e-4

####___________Model Instance______________####
device = torch.device(f'cuda:{7}' if torch.cuda.is_available() else "cpu")
model=fcn(num_classes)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
criterion = nn.NLLLoss2d()

####___________ERROR HANDLING______________####
model = torch.nn.DataParallel(model, device_ids = [7, 1, 3])
#model = torch.nn.DataParallel(model)
print(device)



####___________Training______________####
train_loss_values = []
val_loss_values = []
PATH = "./model-FCN-city.cpt"
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
    # Uploading checkpoint to transfer.sh
    os.system("curl --upload-file ./model-FCN-city.cpt" + str(e) + " http://transfer.sh/model-FCN-city.cpt" + str(e))
    print("\n")
    os.system("rm ./model-FCN-city.cpt" + str(e))
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


