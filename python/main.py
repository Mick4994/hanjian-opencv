from concurrent.futures import process
import cv2
import numpy as np


class Process_ocr:
    def __init__(self):
        self.card_list = []
        self.src_card = []
        self.Template_num = []
        self.split_card_num_list = []
        self.ocr_num = {}

    def TemplateNum(self):
        img = cv2.imread('../OCR credit_card recogition/ocr_a_reference.png',0)
        img = img[20:len(img)-20]
        s_img_list = [[img[i][s] for i in range(len(img))] for s in range(len(img[0]))]
        s_black_pixel_count = [i.count(0) for i in s_img_list]
        in_split = False
        s_split_list = []
        for i in range(len(s_black_pixel_count)):
            if s_black_pixel_count[i] > 0:
                if in_split == False:
                    s_split_list.append(i)
                    in_split = True
            else:
                if in_split == True:
                    s_split_list.append(i)
                in_split = False
        num_split_img_list = []
        for i in range(1,len(s_split_list),2):
            split_img = img[0:-1,s_split_list[i-1]:s_split_list[i]]
            num_split_img_list.append(split_img)
        self.Template_num = num_split_img_list

    def CardProcess(self):
        ocr_card_img = []
        src_card_img = []
        crop_num = []
        for i in range(1,6):
            card_img = cv2.imread('../OCR credit_card recogition/credit_card_0'+str(i)+'.png')
            card_img = cv2.resize(card_img,(1200,750))
            card_img = cv2.blur(card_img,(3,3))
            ocr_img = card_img.copy()
            card_img = cv2.cvtColor(card_img,cv2.COLOR_BGR2GRAY)
            src_card_img.append(card_img)
            _,card_img = cv2.threshold(card_img,150,255,cv2.THRESH_BINARY_INV)
            Rect_x = []
            Rect_area = []
            contours,_ = cv2.findContours(card_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            crop_list = []  
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                area = w*h
                if w>20 and h>45 and w<45 and h<70 and y>300 and y<450:
                    Rect_x.append((x,y,w,h))
                    Rect_area.append(area)
                    cv2.rectangle(ocr_img,(x,y),(x+w,y+h),(0,255,0),2)
                    crop = card_img[y:y+h,x:x+w]
                    ocr_num = self.Compare(crop)
                    cv2.putText(ocr_img,str(ocr_num),(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),1)
                    crop_list.append(crop)
            crop_num.append(crop_list)
            ocr_card_img.append(ocr_img)
        self.split_card_num_list = crop_num
        self.card_list = ocr_card_img

    def Compare(self,card_num):
        Simlarity_list = []
        for num_index in range(len(self.Template_num)):
            num = self.Template_num[num_index]
            card_num = cv2.resize(card_num,(num.shape[1],num.shape[0]))
            _,card_num = cv2.threshold(card_num,1,255,cv2.THRESH_BINARY)
            len_card_num = card_num.reshape(-1)
            len_num = num.reshape(-1)
            Simlarity = len([1 for i in range(len(len_num)) if len_num[i] == len_card_num[i] ])/len(len_num)
            Simlarity_list.append(Simlarity)
        return Simlarity_list.index(max(Simlarity_list))

    def main(self):
        self.TemplateNum()
        self.CardProcess()

if __name__ == '__main__':
    process_ocr = Process_ocr()
    process_ocr.main()
    for img in process_ocr.card_list:
        cv2.imshow('ocr',img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
