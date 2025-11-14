
from hi import process_image


if __name__ == '__main__':

    import PIL
    print(PIL.__version__)

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import time
    import numpy as np
    from torch import nn, optim
    import torch.nn.functional as F
    from torchvision import datasets, transforms, models
    import torchvision
    from collections import OrderedDict
    from torch.autograd import Variable
    from PIL import Image
    from torch.optim import lr_scheduler
    import copy
    import json
    import os
    from os.path import exists
    from tqdm import tqdm
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    #Organizing the dataset
    data_dir = r'C:\Users\jeevan\plantvillage dataset'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/val'
    nThreads = 4
    batch_size = 32
    use_gpu = torch.cuda.is_available()

    """# Label mapping

    You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the plant diseases.
    """

    import json

    with open('categories.json', 'r') as f:
        cat_to_name = json.load(f)

    # Define your transforms for the training and validation sets
    # Data augmentation and normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the datasets with ImageFolder

    data_dir = 'PlantVillage'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    """# Building and training the classifier"""

    # Build and train your network

    # 1. Load resnet-152 pre-trained network
    model = models.resnet152(pretrained=True)
    # Freeze parameters so we don't backprop through them

    for param in model.parameters():
        param.requires_grad = False

    #Let's check the model architecture:
    print(model)

    # 2. Define a new, untrained feed-forward network as a classifier, using ReLU activations

    # Our input_size matches the in_features of pretrained model


    from collections import OrderedDict


    # Creating the classifier ordered dictionary first

    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(2048, 512)),
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(512, 39)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    # Replacing the pretrained model classifier with our classifier
    model.fc = classifier

    #Function to train the model
    def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(1, num_epochs+1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for  inputs, labels in tqdm(dataloaders[phase], desc=f"Epoch {epoch}/{num_epochs} [{phase.upper()}]"):
                    inputs, labels = inputs.to(device), labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            
            # The scheduler step should happen once per epoch
            scheduler.step()
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best valid accuracy: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    # Train a model with a pre-trained network
    num_epochs = 10
    if use_gpu:
        print ("Using GPU: "+ str(use_gpu))
        model = model.cuda()

    # NLLLoss because our output is LogSoftmax
    criterion = nn.NLLLoss()

    # Adam optimizer with a learning rate
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


    model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=10)

    # Now you can add the testing code here
    model_ft.eval()
    accuracy = 0
    model_ft.to(device)
    
    for images, labels in dataloaders['val']:
        images = Variable(images)
        labels = Variable(labels)
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad(): # Use this for inference to save memory
            output = model_ft(images)
            ps = torch.exp(output)
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()
            
    print("Testing Accuracy: {:.3f}".format(accuracy/len(dataloaders['val'])))




    # Save the checkpoint 

    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    model.epochs = num_epochs
    checkpoint = {'input_size': [3, 224, 224],
                    'batch_size': dataloaders['train'].batch_size,
                    'output_size': 39,
                    'state_dict': model.state_dict(),
                    'data_transforms': data_transforms,
                    'optimizer_dict':optimizer.state_dict(),
                    'class_to_idx': model.class_to_idx,
                    'epoch': model.epochs}
    torch.save(checkpoint, 'plants9615_checkpoint.pth')



    # Write a function that loads a checkpoint and rebuilds the model

    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = models.resnet152()
        
        # Our input_size matches the in_features of pretrained model
        input_size = 2048
        output_size = 39
        
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(2048, 512)),
                            ('relu', nn.ReLU()),
                            #('dropout1', nn.Dropout(p=0.2)),
                            ('fc2', nn.Linear(512, 39)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    # Replacing the pretrained model classifier with our classifier
        model.fc = classifier
        
        
        model.load_state_dict(checkpoint['state_dict'])
        
        return model, checkpoint['class_to_idx']

    # Get index to class mapping
    loaded_model, class_to_idx = load_checkpoint('plants9615_checkpoint.pth')
    idx_to_class = { v : k for k,v in class_to_idx.items()}

    """# Inference for classification

    Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the plant disease in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like
    """

    def predict(image_path, model, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        
        # Implement the code to predict the class from an image file
        
        image = torch.FloatTensor([process_image(Image.open(image_path))])
        model.eval()
        output = model.forward(Variable(image))
        pobabilities = torch.exp(output).data.numpy()[0]
        

        top_idx = np.argsort(pobabilities)[-topk:][::-1]
        top_class = [idx_to_class[x] for x in top_idx]
        top_probability = pobabilities[top_idx]

        return top_probability, top_class

    print (predict('PlantVillage/val/Blueberry___healthy/06eacfab-fb39-40e0-bbce-927bc98fa2ac___RS_HL 2663.JPG', loaded_model))

    # Display an image along with the top 5 classes
    def view_classify(img, probabilities, classes, mapper):
        ''' Function for viewing an image and it's predicted classes.
        '''
        img_filename = img.split('/')[-2]
        img = Image.open(img)

        fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
        flower_name = mapper[img_filename]
        
        ax1.set_title(flower_name)
        ax1.imshow(img)
        ax1.axis('off')
        
        y_pos = np.arange(len(probabilities))
        ax2.barh(y_pos, probabilities)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([mapper[x] for x in classes])
        ax2.invert_yaxis()

    #img = 'PlantVillage/val/Apple___Black_rot/0139bc6d-391c-4fd1-bcae-cc74dabfddd7___JR_FrgE.S 2734.JPG'
    #img = 'PlantVillage/val/Tomato___Bacterial_spot/00728f4d-83a0-49f1-87f8-374646fcda05___GCREC_Bact.Sp 6326.JPG'
    img = 'PlantVillage/val/Corn_(maize)___Northern_Leaf_Blight/00a14441-7a62-4034-bc40-b196aeab2785___RS_NLB 3932.JPG'
    #img = 'PlantVillage/val/Apple___healthy/3af9dc00-a64b-4b45-a034-1d190e5277ea___RS_HL 7788.JPG'
    #img = 'PlantVillage/val/Potato___Late_blight/0acdc2b2-0dde-4073-8542-6fca275ab974___RS_LB 4857.JPG'
    #img = 'PlantVillage/val/Tomato___Tomato_Yellow_Leaf_Curl_Virus/0e1fda76-d958-490f-9fcb-21e86c99dbe6___UF.GRC_YLCV_Lab 02200.JPG'

    p, c = predict(img, loaded_model)
    view_classify(img, p, c, cat_to_name)

    img = 'PlantVillage/val/Tomato___Tomato_Yellow_Leaf_Curl_Virus/0e1fda76-d958-490f-9fcb-21e86c99dbe6___UF.GRC_YLCV_Lab 02200.JPG'
    p, c = predict(img, loaded_model)
    view_classify(img, p, c, cat_to_name)

    img = 'PlantVillage/val/Apple___Black_rot/0139bc6d-391c-4fd1-bcae-cc74dabfddd7___JR_FrgE.S 2734.JPG'
    p, c = predict(img, loaded_model)
    view_classify(img, p, c, cat_to_name)

    img = 'PlantVillage/val/Tomato___Bacterial_spot/00728f4d-83a0-49f1-87f8-374646fcda05___GCREC_Bact.Sp 6326.JPG'
    p, c = predict(img, loaded_model)
    view_classify(img, p, c, cat_to_name)

    img = 'PlantVillage/val/Apple___healthy/3af9dc00-a64b-4b45-a034-1d190e5277ea___RS_HL 7788.JPG'
    p, c = predict(img, loaded_model)
    view_classify(img, p, c, cat_to_name)

    """# CONCLUSIONS

    The model can be improved if you change some hyperparameters. You can try using a different pretrained model. It's up to you. Let me know if you can improve the accuracy!
    """

    plt.show()
    
