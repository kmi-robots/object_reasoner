import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

class ImprintedKNet(nn.Module):

    """
    Adding weight imprinting on our implementation of KNet by Zeng et al. (2018)
    - original paper model was in Torch Lua
    https://github.com/andyzeng/arc-robot-vision/tree/master/image-matching
    """

    def __init__(self, feature_extraction=False, num_classes=25):
        super().__init__()

        self.embed = NetForEmbedding(feature_extraction)
        self.embed_prod = NetForEmbedding(feature_extraction=True)

        self.fcs1 = nn.Sequential(

            nn.Linear(2048,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5)

        )

        self.fc2 = nn.Linear(128, num_classes, bias=False) #Removed bias from last layer

        self.scale = nn.Parameter(torch.FloatTensor([10]))


    def forward_once(self, x):
        x = self.embed(x)

        return F.normalize(x)  # #x.view(x.size(0), -1) #self.fc(x.view(x.size(0), -1)) #self.drop(self.linear2(x))

    def forward_prod_branch(self, x):
        x = self.embed_prod(x)
        return F.normalize(x)


    def forward(self, data, trainmode=True):
        if trainmode:
            # Triplet as input
            x0 = self.forward_once(data[:,0,:])
            x1 = self.forward_prod_branch(data[:,1,:]) # feature extracted in prod branch
            x2 = self.forward_once(data[:,2,:])

        else: x0 = self.forward_once(data) # just one test img as input

        res = self.scale * self.l2_norm(self.fcs1(x0))

        if trainmode: return x0, x1, x2, self.fc2(res)
        else: return self.fc2(res)

    def get_embedding(self, x):
        x = self.embed(x)
        return F.normalize(x)

    def extract(self, x):
        x = self.get_embedding(x)
        return self.scale *self.l2_norm(self.fcs1(x))

    def l2_norm(self, x):
        input_size = x.size()
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(x, norm.view(-1, 1).expand_as(x))
        output = _output.view(input_size)

        return output

    def weight_norm(self):

        w = self.linear2.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.linear2.weight.data = w.div(norm.expand_as(w))


class NetForEmbedding(nn.Module):
    """
    Pre-trained Net used to generate img embeddings
    on each siamese pipeline
    """

    def __init__(self, feature_extraction=False):

        super().__init__()
        self.resnet = models.resnet50(pretrained=True)

        #Only drop last FC, keep pre-trained avgpool
        self.mod_resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        if feature_extraction:

            for param in self.mod_resnet.parameters():
                param.requires_grad = False

    def forward(self, x):

        x = self.mod_resnet(x)
        return x.view(x.size(0), -1)



class TripletLoss(nn.Module):

    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean() if size_average else losses.sum()


"""=======================================
Re-implemented from paper by Zeng et al. (2018)
============================================="""

class KNet(nn.Module):

    def __init__(self, feature_extraction=False, num_classes=10):
        super().__init__()

        self.embed = NetForEmbedding(feature_extraction)
        self.embed_prod = NetForEmbedding(feature_extraction=True)

        self.classifier = nn.Sequential(

            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward_once(self, x):
        x = self.embed(x)

        return F.normalize(x)  # #x.view(x.size(0), -1) #self.fc(x.view(x.size(0), -1)) #self.drop(self.linear2(x))

    def forward_prod_branch(self, x):
        x = self.embed_prod(x)
        return F.normalize(x)

    def forward(self, data, trainmode=True):
        if trainmode:
            # Triplet as input
            x0 = self.forward_once(data[:, 0, :])
            x1 = self.forward_prod_branch(data[:, 1, :])  # feature extracted in prod branch
            x2 = self.forward_once(data[:, 2, :])
            return x0, x1, x2, self.classifier(x0)

        else:
            x0 = self.forward_once(data)  # just one test img as input
            return self.classifier(x0)

    def get_embedding(self, x):
        x = self.embed(x)
        return F.normalize(x)


class NNet(nn.Module):
    """same as K-net but without auxiliary classification layer"""

    def __init__(self, feature_extraction=False):
        super().__init__()

        self.embed = NetForEmbedding(feature_extraction)
        self.embed_prod = NetForEmbedding(feature_extraction=True)

    def forward_once(self, x):
        x = self.embed(x)

        return F.normalize(x)  # #x.view(x.size(0), -1) #self.fc(x.view(x.size(0), -1)) #self.drop(self.linear2(x))

    def forward_prod_branch(self, x):
        x = self.embed_prod(x)
        return F.normalize(x)

    def forward(self, data, trainmode=True):
        if trainmode:
            # Triplet as input
            x0 = self.forward_once(data[:, 0, :])
            x1 = self.forward_prod_branch(data[:, 1, :])  # feature extracted in prod branch
            x2 = self.forward_once(data[:, 2, :])
            return x0, x1, x2

        else:
            x0 = self.forward_once(data)  # just one test img as input
            return x0

    def get_embedding(self, x):
        x = self.embed(x)
        return F.normalize(x)


