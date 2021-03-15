import os
from torch.utils.data import DataLoader,Dataset,random_split
import numpy as np
from PIL import Image
from torchvision import transforms


#dataset : InteractiveSegmentation  
#kaggle link : https://www.kaggle.com/4quant/interactivesegmentation

class Interactive_Segmentation_DS(Dataset):
    def __init__(self, img_dir="/interactive_seg/is_ds/images",
                 mask_dir = "/interactive_seg/is_ds/masks",
                 transform = None):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = os.listdir(self.img_dir)
        self.masks = os.listdir(self.mask_dir)


    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):
        #if self.images[index] == self.mask_dir[index]:
        image_path = os.path.join(self.img_dir,self.images[index])
        mask_path = os.path.join(self.mask_dir,self.masks[index])

        #The images are by default RGB, No  need to convert
        image = np.array(Image.open(image_path).convert("RGB"))
        image = Image.fromarray(image)
        # the mask is grayscale. by using "L" in PIL, we convert it to grayscale
        mask = np.array(Image.open(mask_path).convert("L"),dtype=np.float32)  #0.0 for black pixels, and 255.0 for white pixels


        #pre-processing the mask
        mask[mask==255.0] = 1.0  # we want to use sigmoid at the end of our model. that is why we changed it to 1.0
        mask = Image.fromarray(mask)


        if self.transform is not None:
            #augmentations = self.transform(image=image,mask=mask)
            #print("Type o",type(augmentations))

            image = self.transform(image)
            #print("Type o",type(augmentations_img))
            mask = self.transform(mask)
            #image = augmentations["image"]
            #mask = augmentations["mask"]

        #return image,mask
        return image,mask






def get_data_loaders(batch_size=2,
                     image_width = 240,
                     image_height =240,
                     splits=[0.70,0.15,0.15]):
    transform = transforms.Compose([
        transforms.Resize((image_width, image_height)),
        transforms.ToTensor(),
    ])

    dataset = Interactive_Segmentation_DS(transform=transform)
    data_loader_sizes = {}

    train_size = int(len(dataset)*splits[0])

    val_size = int(len(dataset)* splits[1])
    test_size = len(dataset) - (train_size + val_size)

    data_loader_sizes["train"] = train_size
    data_loader_sizes["val"] = val_size
    data_loader_sizes["test"] = test_size

    train_set,val_set,test_set = random_split(dataset,lengths=[train_size, val_size,test_size])
    data_loader_dict = {}
    data_loader_dict["train"] = DataLoader(train_set,
                                           batch_size=batch_size,
                                           shuffle=True)
    data_loader_dict["val"] = DataLoader(train_set,
                                         batch_size=batch_size,
                                         shuffle=False,)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False)


    return data_loader_dict,data_loader_sizes,test_loader

