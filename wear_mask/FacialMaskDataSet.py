import array
import csv
import os,time,cv2,mtcnn,dlib, threading
import random
import numpy as np
import matplotlib as plt
from tool.FileFunction import FileFunction
import tensorflow
from wear_mask import MTCNN



class FacialMaskDataSet:
    """
    create a new dataSet with mask from face without mask
    Include many funciton for face alignment, and work with picture
    MTCNN model
    """
    def __init__(self) -> None:
        self.fileFunction = FileFunction()
        self.count = 0
        self.startTime = time.time()
    def resetStatistic(self):
        self.count = 0
        self.startTime = time.time()
    def face_detection(self, image):
        boundingBoxes = mtcnn.MTCNN().detect_faces(image)
        numberOfFace = len(boundingBoxes)
        if numberOfFace == 1:
            boxCoordinates = list()
            imgSize = np.asarray(image.shape)[0:2]
            rawBoxCoordinates = list()
            for eachBox in boundingBoxes:
                rawBoxCoordinates = eachBox["box"]
                            
            if numberOfFace > 1:
                pass
            else:
                boxCoordinates.append(np.squeeze(rawBoxCoordinates))
                            
            boxCoordinates = np.array(boxCoordinates)
            boxCoordinates = boxCoordinates.astype(np.int)
            return boxCoordinates
        else:
            return None
                

    def alignFace(self,inputDir, outputDir,isDetectMultipleFace=False,outputSize=None,margin=44,datasetRange=None,
                    imgShow=False):
        
        #Currently use code of anohter project with tensorflow v1...
        if tensorflow.__version__.startswith('1.'):
            import tensorflow as tf

        else:
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()

        # Init value for statistic
        startTimeLocal = time.time()
        countLocal = 0
        # colect all folder
        fileFunction = FileFunction()
        dirs = fileFunction.getSubDir(inputDir)
        if len(dirs) <= 0:
            print("There are no sub dictionaries in images")
        else:
            dirs.sort()
            print("Total classified face: ", len(dirs))
            if datasetRange is not None:
                dirs = dirs[datasetRange[0]:datasetRange[1]]
                print("working: from {} to {} ".format(datasetRange[0], datasetRange[1]))
            else:
                print("working: all dataset")
        #----initialization of MTCNN model
        GPU_ratio = 0.6
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        with tf.Graph().as_default():
            config = tf.ConfigProto(log_device_placement=True,
                                    allow_soft_placement=True,
                                    )
            if GPU_ratio is None:
                config.gpu_options.allow_growth = True
            else:
                config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio
            sess = tf.Session(config=config)
            with sess.as_default():
                pnet, rnet, onet = MTCNN.create_mtcnn(sess, None)

        # Handle image for each sub folder
        for dir in dirs:
            pathImages = fileFunction.getPath(dir)
            if len(pathImages) <= 0:
                print("No image in: ", dir)
            else:
                # Create folder to save picture
                save_dir = os.path.join(outputDir, dir.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Work with each image
                countLocal += len(pathImages)
                for path in pathImages:
                    print("work with new image")
                    img = cv2.imread(path)
                    if img is None:
                        print("Cannot read image", path)
                    else:
                        # #Find bounding box v1 run with library
                        # boundingBoxes = mtcnn.MTCNN().detect_faces(img)
                        # numberOfFace = len(boundingBoxes)
                        # if numberOfFace > 0:
                        #     boxCoordinates = list()
                        #     imgSize = np.asarray(img.shape)[0:2]
                        #     rawBoxCoordinates = list()
                        #     for eachBox in boundingBoxes:
                        #         rawBoxCoordinates = eachBox["box"]
                            
                        #     if numberOfFace > 1:
                        #         pass
                        #     else:
                        #         boxCoordinates.append(np.squeeze(rawBoxCoordinates))
                            
                        #     boxCoordinates = np.array(boxCoordinates)
                        #     boxCoordinates = boxCoordinates.astype(np.int)

                        # Find bounding box v2 run with mtcnn have modify
                        boundingBoxes, _ = MTCNN.detect_face(img, minsize, pnet, rnet, onet,
                                                                             threshold, factor)
                        numberOfFace = boundingBoxes.shape[0]
                        if numberOfFace > 0:
                            rawBoxCoordinates = boundingBoxes[:, 0:4]
                            boxCoordinates = []

                            if numberOfFace > 1:
                                pass
                            else:
                                boxCoordinates.append(np.squeeze(rawBoxCoordinates))
                            boxCoordinates = np.array(boxCoordinates)
                            boxCoordinates = boxCoordinates.astype(np.int16)

                            # Crop
                            for i, coordinate in enumerate(boxCoordinates):
                                coordinate = np.squeeze(coordinate)
                                cropCoordinate = np.zeros(4, dtype=int)
                                cropCoordinate[0] = max(coordinate[0] - margin / 2, 0)
                                cropCoordinate[1] = max(coordinate[1] - margin / 2, 0)
                                cropCoordinate[2] = max(coordinate[0] + coordinate[2] + margin / 2, 0)
                                cropCoordinate[3] = max(coordinate[1] + coordinate[3] + margin / 2, 0)
                                croppedImage = img[cropCoordinate[1]:cropCoordinate[3], cropCoordinate[0]:cropCoordinate[2],:]
                                # Resize
                                if outputSize is not None:
                                    croppedImage = cv2.resize(croppedImage, outputSize)
                            
                                # Save
                                fileName = path.split("\\")[-1]
                                fileName = "{}_{}.{}".format(fileName.split(".")[0], str(i), 'jpg')
                                savedPath = os.path.join(save_dir, fileName)
                                cv2.imwrite(savedPath, croppedImage)

                                # Display
                                if imgShow == True:
                                    plt.subplot(1, 2, 1)
                                    plt.imshow(img[:, :, ::-1])
                                    plt.axis("off")

                                    plt.subplot(1, 2, 2)
                                    plt.imshow(croppedImage[:, :, ::-1])
                                    plt.axis("off")

                                    plt.show()
        if countLocal > 0:
            self.count += countLocal
            print("process {} images in {} second".format(countLocal, time.time() - startTimeLocal))
            print("average time to process an image: {}".format((time.time() - startTimeLocal)/countLocal))
        print('done one thread')   
      
    def findMouthCoordinate(self, img:array, detector, predictor):
        x_min = None
        x_max = None
        y_min = None
        y_max = None
        size = None
        landmark = None

        img_rgb = img
        faces = detector(img_rgb, 0)
        if len(faces) == 1:
            for rect in (faces):#coordinate format:[(left,top), (right,bottom)]
                x = list()
                y = list()
                height = rect.bottom() - rect.top()
                width = rect.right() - rect.left()
                landmark = predictor(img_rgb, rect)

                #----get the mouth part
                # In dlib facial landmark mouth offset begin from 48 to 67
                for i in range(48, 68):
                    x.append(landmark.part(i).x)
                    y.append(landmark.part(i).y)

                y_max = min(max(y) + height // 3, img_rgb.shape[0])
                y_min = max(min(y) - height // 3, 0)
                x_max = min(max(x) + width // 3, img_rgb.shape[1])
                x_min = max(min(x) - width // 3, 0)

                size = ((x_max-x_min),(y_max-y_min))#(width,height)

        return x_min, x_max, y_min, y_max, size, landmark

    def attachMaskToFaceDefaul(self, img, imageRandomMask, xMin, yMin, size):
        # Create the filter to clean mask image
        imageRandomMaskRGB = imageRandomMask[:, :, :3]
        imageRandomMaskAlpha = imageRandomMask[:, :, 3]
        _, filterForMask = cv2.threshold(imageRandomMaskAlpha, 200, 255, cv2.THRESH_BINARY)
        imageRandomMaskRGB = cv2.bitwise_and(imageRandomMaskRGB, imageRandomMaskRGB, mask=filterForMask)
        # ----mouth part process: seperate mouth and then add mask to mouth
        mouthPart = img[yMin:yMin + size[1], xMin:xMin + size[0]]
        filterForFace = cv2.bitwise_not(filterForMask)
        mouthPart = cv2.bitwise_and(mouthPart,mouthPart,mask=filterForFace)
        mouthPart = cv2.add(imageRandomMaskRGB,mouthPart)
        # ----addition of mouth and face mask
        img[yMin: yMin + size[1], xMin:xMin + size[0]] = mouthPart

        return img

    def attachMaskToFaceHomoGraphy(self, img, imageRandomMask, coordinatePointsMask, coordinatePointsFace):
        
        matrix,_ = cv2.findHomography(coordinatePointsMask, coordinatePointsFace)

        # Transform mask image base on matrix
        transformMask = cv2.warpPerspective(imageRandomMask, matrix, (img.shape[1], img.shape[0]), None,cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        # Create the filter to clean mask image
        imageMaskRGB = transformMask[:, :, :3]
        imageMaskAlpha = transformMask[:, :, 3]
        _, filterForMask = cv2.threshold(imageMaskAlpha, 200, 255, cv2.THRESH_BINARY)
        imageMaskRGB = cv2.bitwise_and(imageMaskRGB, imageMaskRGB, mask=filterForMask)
        # Add mask image to face
        filterForFace = cv2.bitwise_not(filterForMask)
        img = cv2.bitwise_and(img,img,mask = filterForFace)
        img = cv2.add(imageMaskRGB, img)

        return img

    def createWearMaskDataSet(self, inputDir, outputDir, maskDir, maskAnnotationDir, dataSetRange = None):
       
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
        # Init value for statistic
        startLocalTime = time.time()
        countLocal = 0
        # Read mask pictures
        maskImagePaths = self.fileFunction.getPath(maskDir)
        numberOfMask = len(maskImagePaths)
        if numberOfMask <=0:
            print("there are no mask in {}".format(maskDir))
        else:       
            # colect all folder
            dirs = self.fileFunction.getSubDir(inputDir)
            if len(dirs) <= 0:
                print("There are no sub dictionaries in align_image")
            else:
                # Divide dataset base on range
                dirs.sort()
                print("Total classified face: ", len(dirs))
                if dataSetRange is not None:
                    dirs = dirs[dataSetRange[0]:dataSetRange[1]]
                    print("working: from {} to {} ".format(dataSetRange[0], dataSetRange[1]))
                else:
                    print("working: all dataset")
                
            # Handle image for each sub folder
            for dir in dirs:
                pathImages = self.fileFunction.getPath(dir)
                if len(pathImages) <= 0:
                    print("No image in: ", dir)
                else:
                    countLocal = countLocal + len(pathImages)
                    # Create folder to save picture
                    save_dir = os.path.join(outputDir, dir.split("\\")[-1])
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    # Work with each image
                    
                    #Detect mouth and wear mask
                    for pathImage in pathImages:
                        print("work with new image")
                        img = cv2.imread(pathImage, cv2.IMREAD_UNCHANGED)
                        if img is None:
                            print("read failed {}".format(pathImage))
                        else:
                            xMin, xMax, yMin, yMax, size, landmarks =self.findMouthCoordinate(img,detector,predictor) 
                            if size is not None:

                                # If can find all land mark use Homography to overlay mask
                                landmarks = np.array([[point.x,point.y] for point in landmarks.parts()])
                                if (len(landmarks) != 0) and (landmarks is not None) and (landmarks > 0).all():
                                    offsetRandomMask = random.randint(0, numberOfMask - 1)
                                    pathRandomMask = maskImagePaths[offsetRandomMask]
                                    # pathRandomMask = r"E:\Python Project\FaceMaskRecognize\mask_image\10.png" #test one mask purpose can delete later
                                    imageRandomMask = cv2.imread(pathRandomMask, cv2.IMREAD_UNCHANGED)
                                    coordinatePointsFace = [[point[0], point[1]] for point in landmarks[1:16:1]]
                                    coordinatePointsFace.append([landmarks[29][0],landmarks[29][1]])

                                    nameMask = pathRandomMask.split("\\")[-1].split(".")[0]
                                    pathAnnotation = os.path.join(maskAnnotationDir,nameMask+".csv")
                                    with open(pathAnnotation) as annotationFile:
                                        csvReader = csv.reader(annotationFile)
                                        coordinatePointsMask = []
                                        for row in csvReader:
                                            try:
                                                coordinatePointsMask.append([float(row[1]), float(row[2])])
                                            except ValueError:
                                                continue
                                    coordinatePointsFace = np.array(coordinatePointsFace, dtype = np.float)
                                    coordinatePointsMask = np.array(coordinatePointsMask, dtype=np.float)
                                    img = self.attachMaskToFaceHomoGraphy(img, imageRandomMask, coordinatePointsMask, coordinatePointsFace )
                                    
                                # If cannot find all land mark use default way to overlay mask
                                else:
                                    offsetRandomMask = random.randint(0, numberOfMask - 1)
                                    pathRandomMask = maskImagePaths[offsetRandomMask]
                                    imageRandomMask = cv2.imread(pathRandomMask, cv2.IMREAD_UNCHANGED)
                                    imageRandomMask = cv2.resize(imageRandomMask, size)
                                    img = self.attachMaskToFaceDefaul(img,imageRandomMask, xMin, yMin, size)
                                # -----save img
                                fileName = pathImage.split("\\")[-1]
                                savedPath = os.path.join(save_dir, fileName, )
                                cv2.imwrite(savedPath, img)

        if countLocal > 0:
            self.count += countLocal
            print("process {} images in {} second".format(countLocal, time.time() - startLocalTime))
            print("average time to process an image: {}".format((time.time() - startLocalTime)/countLocal))
        print('done one thread')

    def cleanData(self, numberOfThread, dataSetRange = None):
        # Find total faces
        beginingOffSetFolder = 0
        pathOfFaces = self.fileFunction.getSubDir(os.path.join(os.path.dirname(os.getcwd()),"images"))
        pathOfFaces.sort()
        if(dataSetRange is not None):
            beginingOffSetFolder = dataSetRange[0]
            pathOfFaces = pathOfFaces[dataSetRange[0]: dataSetRange[1]]
        totalFace = len(pathOfFaces)
        facePerThread = totalFace/numberOfThread
        if facePerThread < 1:
            print("Thread is too large")
            return

        # Init param
        inputDir = os.path.join(os.path.dirname(os.getcwd()),"dataset", "lfw")
        outputDir = os.path.join(os.path.dirname(os.getcwd()), "dataset", "lfw_align")
        isDetectMultipleFace = False
        outputSize = None
        margin = 38
        imgShow = False

        # Work with process
        threads = []
        for i in range(numberOfThread):
            #Set up range base on number of face and thread
            dataSetRange = [int(facePerThread*i + beginingOffSetFolder), int(facePerThread*(i+1) + beginingOffSetFolder)]
            if i == numberOfThread - 1:
                dataSetRange[1] = totalFace + beginingOffSetFolder
            # Create thread to clean Face
            t = threading.Thread(target =self.alignFace, args = (inputDir, outputDir, isDetectMultipleFace, outputSize, margin, 
                                dataSetRange, imgShow))
            t.start()
            threads.append(t)
        # Check if no thread alive -> show message 
        for t in threads:
            t.join() 
        if self.count > 0:
            print("process {} images in {} second".format(self.count, time.time() - self.startTime))
            print("average time to process an image: {}".format((time.time() - self.startTime)/self.count))
            print('done clean mask dataset')  
            return
    
    def wearMask(self, numberOfThread, dataSetRange = None):
        # Find total faces
        beginingOffSetFolder = 0
        pathOfFaces = self.fileFunction.getSubDir(os.path.join(os.path.dirname(os.getcwd()),"dataset", "lfw"))
        pathOfFaces.sort()
        if(dataSetRange is not None):
            beginingOffSetFolder = dataSetRange[0]
            pathOfFaces = pathOfFaces[dataSetRange[0]: dataSetRange[1]]
        totalFace = len(pathOfFaces)

        # Init param    
        inputDir = os.path.join(os.path.dirname(os.getcwd()),"dataset", "lfw_align")
        outputDir = os.path.join(os.path.dirname(os.getcwd()), "dataset","lfw_mask")
        maskDir = os.path.join(os.path.dirname(os.getcwd()), "mask_image")
        maskAnnotationDir = os.path.join(os.path.dirname(os.getcwd()), "mask_annotation")
        # Work with process
        if (numberOfThread > 1):
            facePerThread = totalFace/numberOfThread
            if facePerThread < 1:
                print("Thread is too large")
                return

            threads = []
            for i in range(numberOfThread):
                #Set up range base on number of face and thread
                dataSetRange = [int(facePerThread*i + beginingOffSetFolder), int(facePerThread*(i+1) + beginingOffSetFolder)]
                if i == numberOfThread - 1:
                    dataSetRange[1] = totalFace + beginingOffSetFolder
                # Create thread to wear mask
                t = threading.Thread(target =self.createWearMaskDataSet, args = (inputDir, outputDir, maskDir, maskAnnotationDir, dataSetRange))
                t.start()
                threads.append(t)
            # Check if no thread alive -> show message 
            for t in threads:
                t.join()
        else:
            self.createWearMaskDataSet(inputDir,outputDir, maskDir, maskAnnotationDir, dataSetRange)
        if self.count > 0:
            print("process {} images in {} second".format(self.count, time.time() - self.startTime))
            print("average time to process an image: {}".format((time.time() - self.startTime)/self.count))
            print('done create mask dataset')  
            return

def main():
    facialMaskDataSet = FacialMaskDataSet()
    
    # ------------------- Clean Face part-------------------------------------------
    #----------------If dont want to clean face comment this block--------------

    # dataSetRange = None
    # facialMaskDataSet.cleanData(5, dataSetRange)
   
    #----------------------End of align face part--------------------------------------------------
    
    
    # -------------- Create face mask dataset part-------------------------------------------
    # ----------------If dont want to create face mask dataset comment this block--------------
    
    # dataSetRange = None
    # facialMaskDataSet.wearMask(1, dataSetRange)


    path = os.path.join(os.path.dirname(os.getcwd()),"dataset", "lfw_mask")
    print(len(facialMaskDataSet.fileFunction.getPath(path)))

    #----------------------End of Create face mask dataset part--------------------------------------------------

if __name__ == '__main__':
    main()

