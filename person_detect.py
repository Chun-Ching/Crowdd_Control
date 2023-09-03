#%%
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageGrab
import torch
import cv2
import time
import requests

ROOT_DIR = os.path.abspath("C:E:/digi_talent/yolov7-main")#Yolov5資料夾絕對路徑
sys.path.append(ROOT_DIR)

from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import  non_max_suppression, scale_coords
from utils.plots import color_list as colors
#%%

video_path      = 0
video_save_path = ""
# IS_PERSON = False
t1_day = int(time.strftime("%M", time.localtime()))
t1_day_show = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

if __name__ == '__main__' : 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)

    # DATAPATH = 'C:/Users/eddy2/yolov5_new_version/Pokemon/test'
    
    WEIGHTPATH = 'yolov7.pt'
    IS_PERSON = False
    TRIG = False
    # dataset = LoadImages(DATAPATH)
    session = requests.Session()
    model = attempt_load(WEIGHTPATH, map_location = device)
    model.to(device) 

    capture=cv2.VideoCapture(0)
    """
    if video_save_path!="":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),      
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
    """
    fps = 0.0
    person_frame_count = 0
    is_person = 0
    is_not_person = 0
    t2 = time.time()
    while(True):
        t1 = time.time()
        # print(t1)

        if int(time.strftime("%M", time.localtime())) - t1_day >= 1:
            print('a minute passed')
            session.post("http://127.0.0.1:5000/trig", params={"data1": t1_day_show, "data2":time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})
            t1_day = int(time.strftime("%M", time.localtime()))
            t1_day_show = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            

        # img0 = ImageGrab.grab(bbox=(0, 0, 640, 480))
        ret, img0 = capture.read()
        
        img_np = np.array(img0)
        
        # img0 = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
     
        img = letterbox(img_np, new_shape=(320, 320), color=(114, 114, 114))[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, 0.25, 0.45)
        
        for i, det in enumerate(pred):  
            im0 = img0.copy()
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    

        # frame, IS_PERSON = show_pikachu(im0, det, IS_PERSON, t1)
        # ======================================================================================================================
        labels = ['Person']
     
        #b,g,r = cv2.split(img)
        #frame_rgb = cv2.merge((r,g,b))
        #img = frame_rgb
        im0= Image.fromarray(im0)
        
        draw = ImageDraw.Draw(im0)
        
        fontsize = max(round(max(im0.size) / 40), 12)
        font = ImageFont.truetype('text/Arial.ttf', fontsize)
        # fps  = (1./(time.time()-t1))
        # print(fps)
        
        if time.time() - t2 >= 0.5:
            t2 = time.time()
            # print(is_person, is_not_person)
            if is_person > is_not_person * 0.8:
                IS_PERSON = True
            else:
                IS_PERSON = False
            is_person = is_not_person = 0

        if det.size()[0] == 0:
            # print('no person')
            is_not_person += 1
        elif int(det[0][5].cpu().detach().numpy()) != 0:
            # print('no person')
            is_not_person += 1
        else:
            # print('is person')
            
            for info in det:
                is_person += 1
                #color = colors(0)

                target, prob = int(info[5].cpu().detach().numpy()), np.round(info[4].cpu().detach().numpy(), 2)
                
                # print(target)
                if target == 0 and prob >= 0.7:    
                    color = (0, 255, 0)
                    # print('is person!!')
                else:
                    continue
                # elif target == 1:
                #     color = colors(5)
                # elif target == 2:
                #     color = colors(0)
                # elif target == 3:
                #     color = colors(19)
                # elif target == 4:    
                #     color = colors(3)
                
                # color = colors()
            
                # colorr = [color[2], color[1], color[0]]
                # color = tuple(colorr)

                
                #     print(time.time())
                
              
                xmin, ymin, xmax, ymax = info[0], info[1], info[2], info[3]
                if IS_PERSON == False:
                    # print('有人來了')
                    first_pos = (xmax + xmin) / 2
                    # print(f"first_pos: {first_pos}")
                    t3 = time.time()
                    TRIG = True                       
                                     
                IS_PERSON = True

                if TRIG == True:
                    
                    if time.time() - t3 >= 0.5:
                        TRIG = False
                        second_pos = (xmax + xmin) / 2
                        # print(f"second_pos: {second_pos}")
                        if first_pos - second_pos > 0:
                            print('人數 - 1')
                            session.post("http://127.0.0.1:5000/send", params={"data": '-'})
                        else:
                            print('人數 + 1')
                            session.post("http://127.0.0.1:5000/send", params={"data": '+'})
            
                draw.rectangle([xmin, ymin, xmax, ymax], width = 3, outline = color)
                draw.text((xmin, ymin), labels[target] + ':' + str(prob), fill = color, font = font)
            
        # if person_frame_count >= int(fps * 0.8):
        #     IS_PERSON = True
        # else:
        #     IS_PERSON = False
        # print(IS_PERSON)
        # ======================================================================================================================
        frame = np.array(im0)
    
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        
        #print("fps= %.2f"%(fps))
        
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
      
        #print(fps)
        
        cv2.imshow("video",frame)
        c= cv2.waitKey(1) & 0xff 
        if video_save_path!="":
            out.write(frame)
       
        if c==27:
            # img0.release() 
            break
                
    # img0.release()
    # out.release()
    cv2.destroyAllWindows()
            
#--------------------------------------------    


