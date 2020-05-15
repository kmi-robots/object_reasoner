import torch


def imprint(model, device, data_loader, num_classes=41):

    model.eval()
    targets = []
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(data_loader):
            target = torch.max(target, 1)[1]
            input_positive = data[:,0,:].to(device)
            l2norm_emb = model.extract(input_positive)

            if batch_idx == 0:
                output_stack = l2norm_emb
            else:
                output_stack = torch.cat((output_stack, l2norm_emb), 0)
            targets.extend(target)

    target_stack = torch.LongTensor(targets)

    new_weight = torch.zeros(num_classes, 128)
    for i in range(num_classes):

        tmp = output_stack[target_stack == i].mean(0) #Take the average example if more than one is present
        new_weight[i] = tmp / tmp.norm(p=2) #L2 normalize again

    #Use created template/weight matrix to initialize last classification layer
    model.fc2.weight.data = new_weight.to(device)
    model.train()
    return model

def imprint_fortest(model, device, test_data, weight_prior, num_classes=61):
    """
    Similar to above but uses prod data for all classes
    and not based on batch loading
    + reuses weights already learned at training time for known classes
    """
    labels = []
    print(weight_prior.shape)
    print(weight_prior)

    with torch.no_grad():
        for i in range(test_data.prod_data.shape[0]):
            label = test_data.prod_labels[i]
            input_anchor = test_data.prod_data[i,:].unsqueeze(0).to(device)
            l2norm_emb = model.extract(input_anchor)

            if i == 0:
                output_stack = l2norm_emb
            else:
                output_stack = torch.cat((output_stack, l2norm_emb), 0)
            labels.extend(label)

    target_stack = torch.LongTensor(labels)
    new_weight = torch.zeros(num_classes, 128)
    for n in range(num_classes):
        if n <= 40: # for classes seen at training time (the first 41 in ARC case)
            new_weight[n] = weight_prior[n]# use olf weight

        else:
            tmp = output_stack[target_stack == n].mean(0)  # Take the average example if more than one is present
            # tmp = torch.cat((weight_prior[i,:],tmp),0).mean(0)
            new_weight[n] = (tmp / tmp.norm(p=2))  # L2 normalize again

    print(new_weight.shape)
    print(new_weight)
    # Use created template/weight matrix to initialize last classification layer
    model.fc2.weight.data = new_weight.to(device)
    return model
