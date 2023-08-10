import  Model
import  torch
import  utils
import  dataloaders
import  matplotlib.pyplot       as     plt
from torch.utils.data       import DataLoader, Dataset
import os

# Open the file in read mode
with open('Data/Directories.txt', 'r') as file:
    # Read all lines of the file into a list
    lines = file.readlines()

# Remove newline characters from each line and create a list
data = [line.strip() for line in lines]
full_dataset = dataloaders.CIE_Iamges(data,soft_encode=False)

def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    return device

# Select the device
device = select_device()

model = Model.ECCVGenerator().to(device)

model.load_state_dict(torch.load('Model/color_MSE_30_60.pt',map_location=torch.device(device)))
model.eval()


utils.prediction(model_name = "MSE",
                 full_dataset = full_dataset,
                 imagesadresses = data,
                 index = 2226,
                 model = model,
                 device=device)