from tqdm import tqdm
import torch
from torch import nn
from model import UNET
from general_utils import save_predictions_as_imgs,load_checkpoint,save_checkpoint,accuracy
from dataloader2 import get_data_loaders

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 60
# for less computation, I have resizd the images to (240,240)
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
LOAD_MODEL = False



def train_1_epoch(loader,model,optimizer,loss_fn):


    data_loop = tqdm(loader)


    for batch_idx,(data,targets) in enumerate(data_loop):
        data = data.to(DEVICE)
        #targets = targets.float().unsqueeze(1).to(DEVICE)
        targets = targets.float().to(DEVICE)


        #forward prop
        preds = model(data)
        loss = loss_fn(preds,targets)


        #backward Prop.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scaler.scale(loss).backward()
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
       # scaler.update()

        #show the loss value

        data_loop.set_postfix(loss=loss.item())



def main():
   model = UNET()
   model = model.to(DEVICE)
   loss_fn = nn.BCEWithLogitsLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

   data_loader_dict,data_loader_sizes,test_loader= get_data_loaders(batch_size=BATCH_SIZE,
                                                                    image_width=IMAGE_WIDTH,
                                                                    image_height=IMAGE_HEIGHT)


   if LOAD_MODEL:
       load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


   for epoch in range(NUM_EPOCHS):
       train_1_epoch(data_loader_dict["train"],model,optimizer,loss_fn=loss_fn)

       checkpoint = {
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(),}

       save_checkpoint(state=checkpoint)
       accuracy(model,data_loader_dict["val"],device=DEVICE)
       save_predictions_as_imgs(data_loader_dict["val"],model,device=DEVICE,
                             folder="/saved_images/")





if __name__ == '__main__':
    main()









