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

    def __init__(self, feature_extraction=False, norm=True, num_classes=25):
        super().__init__()

        self.embed = NetForEmbedding(feature_extraction)
        self.embed2 = NetForEmbedding(feature_extraction)
        self.embed3 = NetForEmbedding(feature_extraction)

        self.norm = norm

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

    def forward_branch2(self, x):
        x = self.embed2(x)

        return F.normalize(x)

    def forward_branch3(self, x):
        x = self.embed3(x)

        return F.normalize(x)

    def forward(self, data):

        x0 = self.forward_once(data[:,0,:])
        x1 = self.forward_branch2(data[:,1,:])
        x2 = self.forward_branch2(data[:,2,:])

        
        

        import torchvision.transforms as transforms
        x0v = transforms.ToPILImage()(data[0,0,:].cpu())
        x0v.save('positive_prefed.png')
        x1v = transforms.ToPILImage()(x1[0,1,:].cpu())
        x1v.save('anchor_prefed.png')
        x2v = transforms.ToPILImage()(x2[0,2,:].cpu())
        x2v.save('negative_prefed.png')
        import sys
        sys.exit(0)

        res = self.scale * self.l2_norm(self.fcs1(x0))

        return x0, x1, x2, self.fc2(res)

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
