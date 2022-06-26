import cv2
import os
import numpy as np
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import threading
import time
from wear_mask.FaceDetector import FaceDetector
from wear_mask.FaceMaskDetector import FaceMaskDetector
import tensorflow as tf
from train.Classify import Classify
from train.FaceNet import call_instance_FaceNet_with_last_isDense, convert_train_model_to_embedding
from tool.FormatFunction import FormatFunction
from tool.FileFunction import FileFunction
from tool.GlobalValue import GlobalValue
import pickle


class MainApp():
    def __init__(self, video_stream, output_path, model_path, embedding_path, face_detector = None, face_mask_detector = None) -> None:
        #Init variable
        self.is_recognize = True
        self.detect_method = 1 # 0: MTCNN, 1 SSD
        self.margin = 46
        self.global_value = GlobalValue(image_size=[110,110], batch_size = 512, shuffle_size = 1000, ratio_train = 0.8, ratio_test = 0.1, ratio_valid = 0.1, epochs = 40, small_epochs = 50,
                            image_each_class = 15)
        self.format_function = FormatFunction(self.global_value)
        
        # pass param to class
        self.video_stream = video_stream
        self.output_path = output_path
        self.model_path = model_path
        self.embedding_path = embedding_path
        self.face_detector = face_detector
        self.face_mask_detector = face_mask_detector


        #Init model
        self.classify = None
        self.init_model(self.model_path)

        #Init embedding
        self.embedding = self.classify.load_embedding_from_file(self.embedding_path)


        #Create an instance of tkinter frame
        self.frame = None
        self.window = None
        self.pannel_capture = None
        self.pannel_notice = None
        self.capture_button = None
        self.train_button =None
        self.name_text = None
        self.init_main_window()




    
    def init_main_window(self):
        self.window = tk.Tk()
        self.window.title("Face Recognize")
        self.window.geometry("800x800")
        self.window.config(bg="yellow")
        self.window.wm_protocol("WM_DELETE_WINDOW", self.on_close)

        self.pannel_capture = tk.Label(self.window, text = "helo")
        self.pannel_capture.place(relwidth = 1, relheight = 0.9, relx = 0, rely = 0, anchor=tk.NW)

        self.pannel_notice = tk.Label(self.window, text = ".....")
        self.pannel_notice.place(relwidth = 1, relheight = 0.05, relx = 0, rely = 0.8, anchor=tk.NW)

        self.capture_button = tk.Button(self.window, text = "capture", command=self.capture_picture)
        self.capture_button.place(relwidth = 0.2, relheight = 0.05, relx = 0.1, rely = 0.9, anchor=tk.NW)

        self.name_text = tk.Text(self.window, bd = 4)
        self.name_text.place(relwidth = 0.2, relheight = 0.05, relx = 0.4, rely = 0.9, anchor=tk.NW)

        self.capture_button = tk.Button(self.window, text = "train", command=self.train)
        self.capture_button.place(relwidth = 0.2, relheight = 0.05, relx = 0.7, rely = 0.9, anchor=tk.NW)


    def video_loop(self):
        _,self.frame = self.video_stream.read()
        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        # detect face in picture and draw information on image
        if self.is_recognize:
            if self.detect_method == 0:
                bounding_boxes = self.face_detector.detect_one_face(image)
                if len(bounding_boxes) == 1:
                    for bounding_box in bounding_boxes:
                        left,top,right,bottom = self.face_detector.get_coordinate_margin(bounding_box,self.margin, image.shape[1], image.shape[0])
                        face_to_recognize = image[top:bottom,left:right,:]
                        face_to_recognize,_ = self.format_function.process_imagev2(face_to_recognize)
                        label = self.classify.detect_one_image(face_to_recognize, self.embedding, 0.8)
                        image = cv2.rectangle(image,(left,top), (right,bottom), (0,255,0), 3)
                        image = cv2.putText(image, label,(left,bottom), cv2.FONT_ITALIC, 1, (0,255,0), 1, cv2.LINE_AA)
            elif self.detect_method == 1:
                bounding_boxes, is_mask = self.face_mask_detector.detect_face(image)
                if len(bounding_boxes) == 1:
                    for idx,bounding_box in enumerate(bounding_boxes):
                        left = bounding_box[0]
                        top = bounding_box[1]
                        right = bounding_box[2]
                        bottom = bounding_box[3]
                        face_to_recognize = image[top:bottom,left:right,:]
                        face_to_recognize,_ = self.format_function.process_imagev2(face_to_recognize)
                        label = self.classify.detect_one_image(face_to_recognize, self.embedding, 0.8)
                        if is_mask[idx] == 0:#have mask
                            image = cv2.rectangle(image,(left,top), (right,bottom), (0,255,0), 3)
                            image = cv2.putText(image, "mask "+label,(left+10,bottom+30), cv2.FONT_ITALIC, 1, (0,255,0), 2, cv2.LINE_AA)
                        else:#no mask
                            image = cv2.rectangle(image,(left,top), (right,bottom), (255,100,0), 3)
                            image = cv2.putText(image, "no mask "+label,(left+10,bottom+30), cv2.FONT_ITALIC, 1, (255,100,0), 2, cv2.LINE_AA)
        
        # display image into GUI
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.pannel_capture.config(image=image)
        self.pannel_capture.image = image
        self.pannel_capture.after(30, self.video_loop)


    def capture_picture(self):

        #Check condition and find face
        image = self.frame.copy()
        bounding_boxes,_ = self.face_mask_detector.detect_face(image)
        people_name = self.name_text.get("1.0", "end").strip("\n")
        if len(bounding_boxes)== 1 and people_name !="":
            for bounding_box in bounding_boxes:
                left = bounding_box[0]
                top = bounding_box[1]
                right = bounding_box[2]
                bottom = bounding_box[3]
                image = image[top:bottom,left:right,:]

                # Save image to file

                filename = "{}_{}.jpg".format(people_name,time.time())
                dir_path = os.path.join(self.output_path, people_name)
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)
                file_path = os.path.join(dir_path, filename)
                print(file_path)
                cv2.imwrite(file_path, image)
                self.pannel_notice.config(text = "save picture successfully!!!")
                self.pannel_notice.after(800, lambda : self.pannel_notice.config(text = "....."))
    
    def on_close(self):
        self.window.quit()
        self.video_stream.release()

    def show(self):
        self.window.mainloop()
    

    def init_model(self, model_path):
        print("Init model")
        input_size = [self.global_value.IMAGE_SIZE[0], self.global_value.IMAGE_SIZE[1], 3]
        reload_model  = call_instance_FaceNet_with_last_isDense(input_size,10575)
        reload_model.load_weights(model_path)
        embedding_model = convert_train_model_to_embedding(reload_model)
        self.classify = Classify(embedding_model, self.format_function)
        print("Done init model")
        
    
    def train(self):
        self.pannel_notice.config(text = "total data: {}".format(len(self.embedding)))
        database_embedding = self.classify.embedding_all_data_by_directoryV2(self.output_path)
        self.pannel_notice.config(text = "done embedding, waiting for saving")
        for item in database_embedding.items():
            self.embedding[item[0]] = item[1]
        self.classify.save_embedding_to_file(self.embedding, self.embedding_path)
        self.pannel_notice.config(text = "done saving")
        self.pannel_notice.after(800, lambda : self.pannel_notice.config(text = "....."))


def main():
    #Init value
    cap = cv2.VideoCapture(0)
    output_path = os.path.join(os.getcwd(),"data_base_image")
    model_path = os.path.join(os.getcwd(),"models", "model49.h5")
    data_base_path = os.path.join(os.getcwd(), "data_base_encoding","nothing.pkl")
    # data_base_path = os.path.join(os.getcwd(), "data_base_encoding","49_align_encode.pkl")
    face_detector = FaceDetector()
    face_mask_detector = FaceMaskDetector(os.path.join(os.getcwd(), "models","face_mask_detection.pb"))


    app = MainApp(cap, output_path, model_path, data_base_path, face_detector, face_mask_detector)
    app.video_loop()
    app.show()


if __name__ == "__main__":
    main()