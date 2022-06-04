import cv2
from Utils.ocr import detect2
import numpy as np
import PIL.Image as Image



class Stream(object):
    def __init__(self):
        print("Initializing Stream of camera") 
        self.stream = None
        self.text = None
        self.frame = None

        self.start()
    
    def __del__(self):
        self.stream.release()
        cv2.destroyAllWindows()
    
    def start(self):
        print("start streaming of camera")
        self.stream = cv2.VideoCapture(-1)
    
        

    def stop(self):
        self.stream.release()
        cv2.destroyAllWindows()

    def get_text(self):
        return self.text
    
    def save_picture(self):
        ret,image = self.stream.read()
        print(ret)
        if ret == True:
            cv2.imwrite("../static/Images/picture2.jpg",image)
            print("Picture saved!")
            self.stop()
        else:
            print("failed to save picture. Could not read image (self.stream.read())")

    def save_picture_camera(self):
        print("save picture camera function")
        cv2.imwrite("../static/Images/picture1.jpg",self.frame)
        print("Picture saved!")
        self.stop()
    

    
    def load_picture(self,picture):
        print("load picture")
        image = cv2.imread(picture)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ret,jpeg = cv2.imencode(".jpg",image)
        if ret == True:
            return image,jpeg.tobytes()
        else:
            print("failed to read load_picture")
    
    def get_frame(self):
        ret,image = self.stream.read()
        if ret == True:
            ret,jpeg = cv2.imencode(".jpg",image)
            if ret == True:
                return jpeg.tobytes()
        else:
            print("failed to read get_frame")
    
    def get_real_frame(self):
        ret,image = self.stream.read()
        if ret == True:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.text = detect2(image)
            ret,jpeg = cv2.imencode(".jpg",image)
            if ret == True:
                return jpeg.tobytes()
        else:
            print("failed to read get_real_frame")
    
    def get_gray_frame(self):
        ret,image = self.stream.read()
        if ret == True:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            ret,jpeg = cv2.imencode(".jpg",image)
            if ret == True:
                return jpeg.tobytes()
        else:
            print("failed to read get_gray_frame")

    def get_blurr_frame(self):
        ret,image = self.stream.read()
        if ret == True:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.medianBlur(image,5)
            ret,jpeg = cv2.imencode(".jpg",image)
            if ret == True:
                return jpeg.tobytes()
        else:
            print("failed to read get_blurr_frame")


    def get_thresold_frame(self):
        ret,image = self.stream.read()
        if ret == True:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.medianBlur(image,5)
            image = cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            ret,jpeg = cv2.imencode(".jpg",image)
            if ret == True:
                return jpeg.tobytes()
        else:
            print("failed to read get_thresold_frame")


class Webcam():
    def __init__(self):
        self.on = True
    
    def set_switch_webcam(self,switch:bool):
        self.on = switch

    def get_switch_webcam(self):
        return self.on


    def generate(self,stream:Stream,type_image="video_frame"):
        while self.on==True:
            
            if type_image =="gray":
                frame = stream.get_gray_frame()
            elif type_image =="blurr":
                frame = stream.get_blurr_frame()
            elif type_image =="thresold":
                frame = stream.get_thresold_frame()
            elif type_image == "original":
                frame = stream.get_frame()
            elif type_image == "video_frame":
                frame = stream.get_real_frame()

            self.frame = frame

            try:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(frame) + b'\r\n')
            except:
                print("Error - FAILED TO GET FRAME")