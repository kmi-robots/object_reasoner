import torch
from torchvision import transforms
import os
import sys
from utils import img_preproc
import cv2

# Mean and variance values for torchvision modules pre-trained on ImageNet
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
img_w, img_h = 224, 224 # img size required for input to Net


class ImageMatchingDataset(torch.utils.data.Dataset):

    def __init__(self,model, device, args):

        self.device = device
        self.args = args
        

        if self.args.mode == 'train':
            #observed camera imgs were cropped and random hflipped on train
            #cropping done in PIL see utils.img_preproc
            self.trans = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.7),
                        transforms.Resize((img_w, img_h)),
                        transforms.ToTensor()])#,
                        #transforms.Normalize(means, stds)])
            """
            Assumes same format and file naming convention as ARC17 image-matching
            """

            self.data, self.data_emb, self.labels = self.read_files(model,'train-imgs.txt','train-labels.txt') # read all camera training imgs and labels firts
            #change transform for prod imgs
            self.trans = transforms.Compose([
                transforms.Resize((img_w, img_h)),
                transforms.ToTensor()])#,
                #transforms.Normalize(means, stds)])
            self.prod_data, self.prod_emb, self.prod_labels = self.read_files(model,'train-product-imgs.txt','train-product-labels.txt')
            self.triplets, self.final_labels = self.generate_multianchor_triplets(model) # create training triplets based on product images
            print(self.data.shape)
            print(self.labels.shape)
            print(self.prod_data.shape)
            print(self.prod_labels.shape)
            print(self.triplets.shape)
            print(self.final_labels.shape)
        else:
            #just pre-compute embeddings
            self.trans = transforms.Compose([
                transforms.Resize((img_w, img_h)),
                transforms.ToTensor(),
                transforms.Normalize(means, stds)])
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
        # return , target  # triplet data + related label
        #sys.exit(0)
        return self.triplets[index], self.final_labels[index]

    def read_files(self, model, pathtxt, labeltxt,doCrop=False):
        try:
            self.data
        except AttributeError:
            #camera data are being read
            if self.args.mode=='train':
                doCrop=True

        with open(os.path.join(self.args.path_to_arc, pathtxt)) as imgfile, \
            open(os.path.join(self.args.path_to_arc, labeltxt)) as labelfile:
            labels = [torch.LongTensor([int(l)]) for l in labelfile.read().splitlines()]
            imglist = [os.path.join(self.args.path_to_arc, '..', pth) for pth in imgfile.read().splitlines()]

        data = torch.empty((len(imglist), 3, 224, 224))
        embeddings = torch.empty((len(imglist), 2048)) #.cpu() #3, 224, 224))
        #model.cpu()
        #model.eval()
        #import resource
        #soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
        #resource.setrlimit(resource.RLIMIT_AS, (get_memory()*1024/2, hard)) 
        print(str(torch.cuda.memory_allocated(self.device)/(1024*1024*1024)))#[allocated_bytes.all])
        
        for iteration, img_path in enumerate(imglist):
            img_tensor = img_preproc(img_path, self.trans, cropping_flag=doCrop)
            data[iteration, :] = img_tensor
            # Pre-compute embeddings on a ResNet without retrain
            # for all train imgs
            img_tensor = img_tensor.view(1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2]).to(self.device)
        
            try:
                with torch.no_grad():
                     emb =  model.get_embedding(img_tensor) 
                embeddings[iteration, :] = emb
                del img_tensor,emb
                torch.cuda.empty_cache()
            except RuntimeError:
                print(str(torch.cuda.memory_allocated(self.device)/(1024*1024*1024)))#[allocated_bytes.all])
                print(str(iteration))
                import sys
                sys.exit(0)

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


