import torch
import os
from utils import img_preproc

class ImageMatchingDataset(torch.utils.data.Dataset):

    def __init__(self,model, device, args, transforms):

        self.device = device
        self.args = args
        self.trans = transforms

        """
        Assumes same format and file naming convention as ARC17 image-matching
        """
        if self.args.mode == 'train':
            self.data, self.data_emb, self.labels = (self.read_files(model,'train-imgs.txt','train-labels.txt')) # read all camera training imgs and labels firts
            self.prod_data, self.prod_emb, self.prod_labels = (self.read_files(model,'train-product-imgs.txt','train-product-labels.txt'))
            self.triplets, self.final_labels = self.generate_multianchor_triplets(model) # create training triplets based on product images
        else:
            #just pre-compute embeddings
            self.data, self.data_emb, self.labels = (self.read_files(model, 'test-imgs.txt','test-labels.txt'))
            self.prod_data, self.prod_emb, self.prod_labels = (self.read_files(model, 'test-product-imgs.txt','test-product-labels.txt'))

        return

    def __len__(self):
        # used by the data loader
        # implemented only for training data here
        return len(self.triplets)

    def __getitem__(self, index):
        # used by the data loader
        # implemented only for training data here
        imgs, target = self.triplets[index], self.final_labels[index]

        return [imgs[i] for i in range(len(imgs))], target  # triplet data + related label

    def read_files(self, model, pathtxt, labeltxt):

        with open(os.path.join(self.args.path_to_arc, pathtxt)) as imgfile, \
            open(os.path.join(self.args.path_to_arc, labeltxt)) as labelfile:
            labels = [torch.LongTensor([int(l)]) for l in labelfile.read().splitlines()]
            imglist = [os.path.join(self.args.path_to_arc, '..', pth) for pth in imgfile.read().splitlines()]

        data = torch.empty((len(imglist), 3, 224, 224))
        embeddings = torch.empty((len(imglist), 2048)) #3, 224, 224))
        for iteration, img_path in enumerate(imglist[:10]):
            img_tensor = img_preproc(img_path, self.trans)
            data[iteration, :] = img_tensor
            # Pre-compute embeddings on a ResNet without retrain
            # for all train imgs
            #img_tensor = img_tensor.view(1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
            # embeddings[iteration, :] = model.get_embedding(img_tensor.to(self.device))

        return data, embeddings, torch.stack(labels)

    def generate_multianchor_triplets(self, model):


        triplet_data = []
        labels = []
        for i in range(self.data.shape[0]):
            # for each camera/real-world image embedding
            current_e = self.data_emb[i, :]
            positive = self.data[i,:]
            # pick closest product img of same class as anchor
            all_distances = torch.matmul(current_e, self.prod_emb.t())#self.prod_data.t()) # embeddings are normalised, so cosine similarity reduces to matrix/vector multiplication
            prod_ranking, ranking_indices = torch.sort(all_distances, descending=True)
            for k in range(prod_ranking.shape[0]):
                kNN_index = ranking_indices[k].item()
                if self.prod_labels[kNN_index].item() == self.labels[i].item():
                    #   pick comes from same class as input, choose its original vector as anchor
                    anchor = self.prod_data[kNN_index, :]
                    break
            # and then pick closest train/camera/real-world img from different class as negative example
            # i.e., hardest one to disambiguate
            # [The approach by Zeng et al. uses a random pick from different class instead]
            all_rgb_distances = torch.matmul(current_e, self.data_emb.t()) #self.data.t())
            rgb_ranking, rgb_indices = torch.sort(all_rgb_distances, descending=True)
            #Note: picking from different class automatically excludes the embedding itself (most similar to itself)
            for k in range(rgb_ranking.shape[0]):
                kNN_index = rgb_indices[k].item()
                if self.labels[kNN_index].item() != self.labels[i].item():
                    negative = self.data[kNN_index, :]
                    break

            triplet_data.append(torch.stack([positive, anchor, negative]))
            #And class label as one-hot encoding among the N known classes
            temp = torch.zeros(self.args.numobj)
            temp[self.labels[i].item()-1] = 1  #labels in range 1-41, indices in range 0-40
            labels.append(temp)
        print("Image triplets formed")
        return torch.stack(triplet_data), torch.stack(labels)

