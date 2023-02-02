import os
import torchvision
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tool.FileFunction import FileFunction
from train_pytorch import config
import pickle
from matplotlib import pyplot as plt


class CustomImageDataset(Dataset):
    def __init__(self, image_paths, label_dict, transform = None, target_transform = None):
        self.image_paths = image_paths
        self.label_dict = label_dict
        self.transform = transform
        self.target_transform = target_transform
    
    
    def __len__(self):
        return len(self.image_paths)

    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = read_image(image_path)
        label = image_path.split(os.sep)[-2]
        if self.label_dict:
            label = self.label_dict[label]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)
        return image, label

def main():
    file_function = FileFunction()
    #Read label dictionary(name of people not the path of image)
    path = os.path.join(os.path.dirname(os.getcwd()),"cache","data","label_dict.pkl")
    with open(path, 'rb') as f:
        label_dict = pickle.load(f)
    # Get list path images
    path = os.path.join(os.path.dirname(os.getcwd()),"cache","data","path_image_no_mask.pkl")
    with open(path, 'rb') as f:
        path_image_no_mask = pickle.load(f)
        path_image_no_mask = file_function.get_data_path_with_limit(path_image_no_mask,15)
    path = os.path.join(os.path.dirname(os.getcwd()),"cache","data","path_image_mask.pkl")
    with open(path, 'rb') as f:
        path_image_mask = pickle.load(f)
        path_image_mask = file_function.get_data_path_with_limit(path_image_mask,15)
    path_image_no_mask.extend(path_image_mask)
    
    # push paths to DataLoader
    transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(config.SIZE),
             torchvision.transforms.ConvertImageDtype(torch.float)])
    train_dataset = CustomImageDataset(path_image_no_mask,label_dict, transform =transform)
    train_data_loader = DataLoader(train_dataset, 96, shuffle = True)
    for data in train_data_loader:
        images, labels = data
        plt.imshow(torchvision.transforms.ToPILImage()(images[0]))
        print(labels[0])
        plt.show()
        break
if __name__ == "__main__":
    main()