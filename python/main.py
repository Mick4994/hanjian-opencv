import cv2

class Process_ocr:
    def __init__(self):
        self.src_img = []
        self.bin_img = []
        self.split_img = {}
        self.ocr_num = {}

    def GetTemplateNum(self):
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
        return num_split_img_list

    def ReadCreditCard(self):
        card_img_list = []
        for i in range(1,6):
            card_img = cv2.imread('../OCR credit_card recogition/credit_card_0'+str(i)+'.png',0)
            card_img = cv2.resize(card_img,(1200,750))
            card_img = cv2.blur(card_img,(3,3))
            _,bin_card = cv2.threshold(card_img,150,255,cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(bin_card,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            Rect_x = []
            Rect_area = []
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                area = w*h
                if w>400:
                    Rect_x.append((x,y,w,h))
                    Rect_area.append(area)
                    crop = bin_card[y:y+h,x:x+w]
                    crop = cv2.resize(crop,(1200,750))
                    _,crop = cv2.threshold(crop,0,255,cv2.THRESH_BINARY)
            card_img_list.append(crop)

    def ThreshImg(self):
        _,self.bin_img = cv2.threshold(self.src_img,150,255,cv2.THRESH_BINARY_INV)

    def SplitImg(self):
        black_pixel_count = [i.tolist().count(0) for i in self.src_img]
        in_split = False
        h_split_list = []
        #分割纵向
        for i in range(len(black_pixel_count)):
            if black_pixel_count[i] > 5:
                if in_split == False:
                    h_split_list.append(i)
                    in_split = True
            else:
                if in_split == True:
                    h_split_list.append(i)
                in_split = False
        h_split_img_list = []
        h_split_start = []
        for i in range(1,len(h_split_list),2):
            h_split_img = self.src_img[h_split_list[i-1]:h_split_list[i]]
            h_split_start.append(h_split_list[i-1])
            h_split_img_list.append(h_split_img)
        #分割横向
        for j in range(len(h_split_img_list)):
            s_img_list = []
            for s in range(len(h_split_img_list[j][0])):
                s_img = []
                for i in range(len(h_split_img_list[j])):
                    s_img.append(h_split_img_list[j][i][s])
                s_img_list.append(s_img)
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
            for i in range(1,len(s_split_list),2):
                split_img = h_split_img_list[j][0:-1,s_split_list[i-1]:s_split_list[i]]
                position = [(s_split_list[i-1],h_split_start[j]),(s_split_list[i],h_split_start[j]+len(split_img))]
                self.split_img.update({str(position):split_img})

    def MatchTemplate(self,Template_num):
        for box,split_img in self.split_img:
            resize_img = cv2.resize(split_img,Template_num[0].shape)
            for num_img in Template_num:
                res = cv2.matchTemplate(resize_img,num_img,cv2.TM_SQDIFF_NORMED)
                w,h = num_img.shape[::-1]
                cv2.normalize( res, res, 0, 1, cv2.NORM_MINMAX, -1 )
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                res_img = self.src_img.copy()
                cv2.rectangle(res_img,min_loc,(min_loc[0]+w,min_loc[1]+h) ,(255,0,0),2)
    def main(self):
        Template_num = self.GetTemplateNum()
        self.ThreshImg()
        self.SplitImg()