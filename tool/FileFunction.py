# This includes function to work with file

import os
import shutil
import numpy as np
import cv2
import dlib
from imutils import face_utils
import random
import pickle
from PIL import Image
from tqdm import tqdm

class FileFunction:

    """_summary_ A class include all function to work with file
                for example: read file, get path
    """
    def getPath(self, dir:str):
        """_summary_:
             Get all file with given expansion(set in this function)

        Args:
            dir (str): _description_ name of sub folder that will be read 

        Returns:
            _type_: _description_ A list include string path of file
        """
        image_extensions = {'jpg', 'png', 'jpeg'}
        paths = []
        for dirname, sub_dirname, filenames in os.walk(dir):
            if len(filenames) > 0:
                for filename in filenames:
                    if filename.split(".")[-1] in image_extensions:
                        full_path = os.path.join(dirname, filename)
                        paths.append(full_path)
        return paths
    def getPathAllPictureInProject(self):
        root_dir = os.path.join(os.getcwd(),"..","images")
        return self.getPath(root_dir)

    def getSubDir(self, dir:str):
        """ 
        Get all sub folder in the given dictionary of project

        Args:
            dir (str): _description_
        """
        paths = list()
        for item in os.scandir(dir):
            if item.is_dir():
                full_path = os.path.join(dir, item)
                paths.append(full_path)
        return paths        
    def getAllSubDirInImages(self):
        root_dir = os.path.join(os.getcwd(),"..","images")
        return self.getSubDir(root_dir)

    def dataDistribution(self, dir:str, ratio=0.7):
        paths = self.getPath(dir)
        leng = len(paths)
        if leng <= 0:
            print("No supported image files in {}".format("img_dir"))
        else:
            # ----shuffle
            paths = np.array(paths) 
            indice = np.random.permutation(leng)
            paths = paths[indice]

            # ----distribution
            num = int(leng * ratio)
            part_1 = paths[:num]
            part_2 = paths[num:]

            # ----create new directories
            directory_1 = os.path.join(dir, "ratio_" + "%.2f" % ratio)
            directory_2 = os.path.join(dir, "ratio_" + "%.2f" % (1-ratio))

            for dir_path in [directory_1, directory_2]:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

            #----copy and save
            # ====the ratio part
            for path in part_1:
                new_file_path = path.split("\\")[-1]
                new_file_path = os.path.join(directory_1, new_file_path)
                if not os.path.exists(new_file_path):
                    shutil.copy(path, new_file_path)

            # ====the (1 - ratio) part
            for path in part_2:
                new_file_path = path.split("\\")[-1]
                new_file_path = os.path.join(directory_2, new_file_path)
                if not os.path.exists(new_file_path):
                    shutil.copy(path, new_file_path)

    def dataDistributionForProject(self):
        """
        Distribute data for this project, device raw data into 2 subfolder
        Step 1: get path to "images" folder
        Step 2: get path to all sub folder in image
        Step 3: for each one call function dataDistribution to divide data into two subset
        """
        root_dir = os.path.join(os.getcwd(),"..","images")
        for subDirectory in self.getSubDir(root_dir):
            self.dataDistribution(dir = subDirectory)
        print("done")

    def deleteTrainingAndTestFolderForProject(self):
        """
        Delete all training and testing data folder in project 
        Step 1: Search All folder in images
        Step 2: Search all sub folder of the folder above
        Step 3: delete all subfolder
        """
        root_dir = os.path.join(os.getcwd(),"..","images")
        for subDirectory in self.getSubDir(root_dir):
            for needToBeDeletedDirectory in self.getSubDir(subDirectory):
                try:
                    shutil.rmtree(needToBeDeletedDirectory)
                except OSError as e:
                     print("Error: %s - %s." % (e.filename, e.strerror))
        print("Delete All train and test data")
    def openCamera(self):
        cap = cv2.VideoCapture(cv2.CAP_DSHOW)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('../src/models/shape_predictor_68_face_landmarks.dat')
        while(True):
            # capture fram by frame
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # show the face number
                cv2.putText(gray, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for (index, (x, y)) in enumerate(shape):
                    cv2.circle(gray, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(gray, str(index), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            list68LandMark = list()
            cv2.imshow("frame", gray)
            if(cv2.waitKey(1) == ord("q")):
                break

        cap.release()
        cv2.destroyAllWindows()

    def get_data_path_with_limit(self, input_list:list, limit:int):
        datapath = list()
        for small_list in input_list:
            random.shuffle(small_list)
            min = np.minimum(len(small_list), limit)
            datapath.extend(small_list[:min])
        return datapath
    
    def get_data_path_by_dictionary(self, dir:str):
        datapath = list()
        folder_each_class = self.getSubDir(dir)
        for each_folder in folder_each_class:
            image_in_folder = self.getPath(each_folder)
            random.shuffle(image_in_folder)
            datapath.append(image_in_folder)
        return datapath

    def detect_and_remove_error_image(self, directory):
        list_image = self.getPath(directory)
        bads = []
        for image_path in tqdm(list_image):
            try:
                img = Image.open(image_path)
                img.verify()
            except (IOError, SyntaxError) as e:
                bads.append(image_path)
                os.remove(image_path)
        for bad_path in bads:
            print(bad_path)
    def clear():
        os.system('cls')


def main():
    fileFunction = FileFunction()
    path = os.path.join(os.path.dirname(os.getcwd()), "dataset", "lfw_mask")
    print(len(fileFunction.getPath(path)))

    


if __name__ == '__main__':
    main()
