import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import numpy as np
from models import TripletLoss


def train(model, device, train_loader, epoch, optimizer, num_epochs, metric_avg ='weighted'):

    """
    Defaults to weighted because multi-class classification in K-net
    But can be set to binary for N-net or other binary classifiers
    """

    model.train()
    criterion = TripletLoss()

    # accurate_labels = 0
    all_labels = 0
    running_loss = 0
    labels = []
    predictions = []

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        target = target.type(torch.LongTensor).to(device)

        target = torch.max(target, 1)[1]
        emb_a, emb_p, emb_n, output_logits = model(data.to(device))
        classif_loss = F.cross_entropy(output_logits, target)

        triplet_loss = criterion(emb_a, emb_p, emb_n)
        loss = classif_loss + triplet_loss

        norm_tloss = emb_a.shape[0] * triplet_loss.item()
        norm_closs = output_logits.shape[0] * classif_loss.item()
        running_loss += norm_tloss + norm_closs
        #Multi-class instead of binary
        predictions.extend(torch.argmax(output_logits, dim=1).tolist())
        labels.extend(target.tolist())
        # accurate_labels = torch.sum(torch.argmax(output_logits, dim=1) == target).cpu()
        all_labels = all_labels + len(target)

        loss.backward()
        optimizer.step()

    accuracy = accuracy_score(labels, predictions)
    epoch_loss = running_loss / all_labels

    # Compute epoch-level metrics with sklearn
    p, r, f1, sup = precision_recall_fscore_support(np.asarray(labels), np.asarray(predictions), average=metric_avg)
    print("Epoch {}/{}, Loss: {:.6f}, Accuracy: {:.6f}%, Precision: {:.6f}, Recall: {:.6f}".format(epoch + 1, num_epochs, epoch_loss, accuracy, p, r))

