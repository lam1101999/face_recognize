from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import math
from matplotlib import pyplot as plt
import shutil
from tqdm import tqdm


def image_removal_by_embedding(removal_dir, mislabel_dir, threshold = 1, delete_range = None):
    image_format = ["jpg", "png"]
    # Init model
    model = InceptionResnetV1(pretrained='vggface2').eval()
    IMAGE_SIZE = [160,160]
    BATCH_SIZE = 96
    
    # Collect list of folder inside removal_dir
    dir_paths = [obj.path for obj in os.scandir(removal_dir) if obj.is_dir()]
    dir_paths.sort()
    if delete_range != None:
        dir_paths = dir_paths[delete_range[0]:delete_range[1]]
    
    
    # Iterate through each folder(class)
    for dir_path in tqdm(dir_paths):
        # Read all image inside each folder
        image_paths = np.array([file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in image_format])
        len_paths = len(image_paths)
        if len_paths<=1:
            continue
        # Calculate embedding of all image in image_paths
        ites = math.ceil(len_paths/BATCH_SIZE)
        embeddings = np.zeros([len_paths,512], dtype = np.float32)
        for idx in range(ites):
            offset_start = idx*BATCH_SIZE
            offset_end = np.minimum(len_paths, offset_start + BATCH_SIZE)
            # ----read batch data
            batch_data = convert_list_path_to_matrix_image(image_paths[offset_start: offset_end], IMAGE_SIZE)
            embeddings[offset_start: offset_end] = model(batch_data).detach()
        # Calculate average distance from one image to all other images
        distance_matrix = np.zeros([len_paths,1])
        for row, embedding in enumerate(embeddings):
            distance = np.sum(np.sqrt(np.sum((embedding - embeddings)**2, 1)))/(len_paths-1)
            distance_matrix[row] = distance
        index_image_should_be_deleted = np.array([idx for idx, distance in enumerate(distance_matrix) if distance > threshold])
        # Create folder then move image have average distance larger than threshold
        list_path_image_should_be_deleted = np.array([])
        if index_image_should_be_deleted.shape[0] > 0:
            list_path_image_should_be_deleted = image_paths[index_image_should_be_deleted]
            
        if list_path_image_should_be_deleted.shape[0] > 0:
            index_deleted_image = 0
            class_name = list_path_image_should_be_deleted[0].split(os.sep)[-2]
            print("-----------Detect mislabel image------------", class_name)
            folder_save_mislabel = os.path.join(mislabel_dir, class_name)
            if not os.path.exists(folder_save_mislabel):
                os.makedirs(folder_save_mislabel)
            for idx,path in enumerate(list_path_image_should_be_deleted):
                print(path,"dis: ", distance_matrix[index_image_should_be_deleted[index_deleted_image]])
                index_deleted_image+=1
                shutil.copy2(path, folder_save_mislabel)
                os.remove(path)

def convert_list_path_to_matrix_image(list_path, size):
    transform = transforms.ToTensor()
    list_tensor_image = []
    
    for idx,path in enumerate(list_path):
        image = Image.open(path).convert("RGB")
        image = image.resize(size)
        tensor_image = transform(image)
        list_tensor_image.append(tensor_image)
    
    tensor_of_tensor_image = torch.stack(list_tensor_image,0)
    return tensor_of_tensor_image

def main():
    removal_dir = os.path.join(os.path.dirname(os.getcwd()),"dataset","CASIA_align")
    mislabel_dir = os.path.join(os.path.dirname(os.getcwd()),"dataset","CASIA_align_mis")
    delete_range = [8000,10576]
    image_removal_by_embedding(removal_dir, mislabel_dir, threshold = 1.1, delete_range = delete_range)

if __name__ == "__main__":
    main()