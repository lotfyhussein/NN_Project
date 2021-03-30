from torch.utils.data import Dataset, DataLoader
from dataset import cityscape_dataset
import torch
from network import fcn
from torch.autograd import Variable
import random
import torch.nn as nn
from dataset import get_loader 
from tqdm import tqdm
import numpy as np
from denseCRF import dense_crf
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,classification_report,jaccard_score,accuracy_score

####___________Loading Dataset______________####
num_classes=20
valid_loader, valid_dataset = get_loader('val')


####___________Model Instance______________####
device = torch.device(f'cuda:{6}' if torch.cuda.is_available() else "cpu")
model=fcn(num_classes)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)
criterion = nn.NLLLoss2d()
model = torch.nn.DataParallel(model, device_ids = [6, 1, 3])

####___________ERROR HANDLING______________####
#model = torch.nn.DataParallel(model)
##


####___________Loading checkpoint__________####
checkpoint = torch.load("./model-FCN-city.cpt60", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
print(device)
model.eval()

##Confusion Matrix Metrics
true_labels = []
predictions = []

#Preparing predictions and labels for evaluation
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

######################################################
# Dense CRF Pos-processing 
######################################################
print("Applying DenseCRF post processing......")
crf_processed = []
for i in tqdm(range(len(valid_dataset))):
  test_data, test_label = valid_dataset[i]
  img = Variable(test_data.unsqueeze(0)).to(device)
  output = model(img)
  softmax = nn.Softmax(dim=1)
  output = softmax(output)
  output.data.cpu().numpy()
  crf_output = np.zeros(output.shape)
  images = img.data.cpu().numpy().astype(np.uint8)
  for i, (image, prob_map) in enumerate(zip(images, output)):
      image = image.transpose(1, 2, 0)
      crf_output[i] = dense_crf(image, prob_map.cpu().detach().numpy())
  output = crf_output
  N,_,h,w=output.shape
  pred=output.transpose(0,2,3,1).reshape(-1,num_classes).argmax(axis=1).reshape(N,h,w)
  crf_processed.append(pred)


# Applying all metrics again on the pos-processed prediction
true_labels = []
predictions = []
for i in tqdm(range(len(valid_dataset))):
  im , lbl = valid_dataset[i]
  im = Variable(im.unsqueeze(0)).to(device)
  out = model(im)
  pred = out.max(1)[1].squeeze().cpu().data.numpy()
  true_labels.extend(lbl.flatten())
  predictions.extend(crf_processed[i].flatten())

print("Recall,F1 Score, and Precision:")
print(classification_report(true_labels,predictions,digits=3))

print("Jaccard Score:")
print(jaccard_score(true_labels, predictions, average=None))
print(jaccard_score(true_labels, predictions, average="macro"))

print("Accuracy Score:")
print(accuracy_score(true_labels, predictions, normalize=True))

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
