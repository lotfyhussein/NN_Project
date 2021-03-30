from torch.utils.data import Dataset, DataLoader
from dataset import cityscape_dataset
import torch
from network import R2U_Net
from torch.autograd import Variable
import random
import torch.nn as nn
from dataset import get_loader 
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,classification_report,jaccard_score,accuracy_score

####______________Loading Dataset____________###
num_classes=20
valid_loader, valid_dataset = get_loader('val')

####______________Hyperparameters____________###
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

####___________ERROR HANDLING______________####
#model = torch.nn.DataParallel(model)
##

####______________Loading checkpoint___________###
checkpoint = torch.load("./model-R2U-Net.cpt43", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
print(device)
model.eval()

##Confusion Matrix Metrics
true_labels = []
predictions = []

for i in tqdm(range(len(valid_dataset))):
  im , lbl = valid_dataset[i]
  im = Variable(im.unsqueeze(0)).to(device)
  out = model(im)
  pred = out.max(1)[1].squeeze().cpu().data.numpy()
  true_labels.extend(lbl.flatten())
  predictions.extend(pred.flatten())



#Senstivity (Recall), F1 Score, and Precision
print("Recall,F1 Score, and Precision:")
print(classification_report(true_labels,predictions,digits=3))

#Jaccard Similarity
print("Jaccard Score:")
print(jaccard_score(true_labels, predictions, average=None))
print(jaccard_score(true_labels, predictions, average="macro"))

#Accuracy
print("Accuracy Score:")
print(accuracy_score(true_labels, predictions, normalize=True))

#Specificity
CM = confusion_matrix(true_labels,predictions)

false_positives = []
true_negatives = []

for l in range(len(CM)):
    class_true_negative = 0
    class_false_positive = 0
    for i,r in enumerate(CM):
        for j,v in enumerate(r):
            if i != l and j != l:
                class_true_negative+=v
            if i != l and j == l:
                class_false_positive+=v                        
    true_negatives.append(class_true_negative)
    false_positives.append(class_false_positive)

specifity = np.array(true_negatives)/(np.array(true_negatives)+np.array(false_positives))

print("Specificity:")
print(specifity)
