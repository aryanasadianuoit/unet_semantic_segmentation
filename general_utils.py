import torch
import torchvision
#from dataloader2 import Interactive_Segmentation_DS
#from torch.utils.data import DataLoader



def save_checkpoint(state, filename="/home/aasadian/interactive_seg/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model,path="/home/aasadian/interactive_seg/"):
    print("=> Loading checkpoint")
    model.load_state_dict(path+checkpoint["state_dict"])



def accuracy(model,data_loader,device):


    num_correct_pixels = 0
    num_total_pixels = 0
    dice_score = 0

    model.eval()

    with torch.no_grad():

        for image,target in data_loader:

            image = image.to(device)

            #we use unsqueeze(1), since the graysclae images do not have channel
            #target = target.unsqueeze(1).to(device)
            target = target.to(device)


            # The model does not an activation function at the end
            preds = torch.sigmoid(model(image))

            #for binary classification, if the preds > 0.5 ==> consider as 1, otherwise 0
            preds = (preds > 0.5).float()

            num_correct_pixels += (preds == target).sum()  # sum over all the pixels
            num_total_pixels += torch.numel(preds)
            dice_score += (2 * (preds * target).sum()) / (
                    (preds + target).sum() + 1e-8
            )

    print(f"Validation Acc ===> {(num_correct_pixels / num_total_pixels) * 100:.2f}   Dice Score : {dice_score/len(data_loader):.2f}")

    model.train()


def save_predictions_as_imgs(
    loader, model, folder="/home/aasadian/interactive_seg/preds/", device="cuda"):
    model.eval()
    for idx, (image, target) in enumerate(loader):
        image = image.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(image))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(target, f"{folder}{idx}.png")

    model.train()
