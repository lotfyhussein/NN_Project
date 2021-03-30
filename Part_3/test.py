from sklearn.metrics import confusion_matrix
import numpy as np

true_labels = [0,1,2,0,1,1,2]
predictions = [0,1,1,0,2,0,2]

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

#print(CM)
print(false_positives)
print(true_negatives)

specifity = np.array(true_negatives)/(np.array(true_negatives)+np.array(false_positives))
print(specifity)
