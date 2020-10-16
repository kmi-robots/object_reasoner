import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from models import TripletLoss


def validate(args, model, device, val_loader, optimizer, early_stopper, metric_avg ='weighted'):

    """
    validation module
    Defaults to weighted because multi-class classification in K-net
    But can be set to binary for N-net or other binary classifiers
    """

    model.eval()
    criterion = TripletLoss()

    with torch.no_grad():

        # accurate_labels = 0
        all_labels = 0
        running_loss = 0
        labels = []
        predictions = []

        for batch_idx, (data, target) in enumerate(val_loader):

            optimizer.zero_grad()
            target = target.type(torch.LongTensor).to(device)

            if args.model =='k-net' or args.model =='imprk-net':
                target = torch.max(target, 1)[1]
                emb_p, emb_a, emb_n, output_logits = model(data.to(device))
                classif_loss = F.cross_entropy(output_logits, target)
            elif args.model =='n-net':
                emb_p, emb_a, emb_n = model(data.to(device))
                classif_loss = 0.0

            triplet_loss = criterion(emb_a, emb_p, emb_n)
            norm_tloss = emb_a.shape[0] * triplet_loss.item()

            if args.model =='n-net':
                running_loss += norm_tloss
                all_labels = all_labels + data[0].shape[0]  # Cannot really classify in this case

            elif args.model =='k-net' or args.model =='imprk-net':
                norm_closs = output_logits.shape[0] * classif_loss.item()
                running_loss += norm_tloss + norm_closs
                #Multi-class instead of binary
                predictions.extend(torch.argmax(output_logits, dim=1).tolist())
                labels.extend(target.tolist())
                # accurate_labels = torch.sum(torch.argmax(output_logits, dim=1) == target).cpu()
                all_labels = all_labels + len(target)

        accuracy = accuracy_score(labels, predictions)
        epoch_loss = running_loss / all_labels

        if args.model =='n-net':
            # No classification layer, only loss can be computed
            print("Validation Loss: {:.6f}, Accuracy: {:.6f}%, Precision: {:.6f}, Recall: {:.6f}".format(epoch_loss,0.0, 0.0, 0.0))
        else:
            # Compute epoch-level metrics with sklearn
            p, r, f1, sup = precision_recall_fscore_support(np.asarray(labels), np.asarray(predictions), average=metric_avg)
            print("Validation Loss: {:.6f}, Accuracy: {:.6f}%, Precision: {:.6f}, Recall: {:.6f}".format(epoch_loss, accuracy, p, r))

        return early_stopper(epoch_loss,model)
