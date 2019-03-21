import numpy as np
import cv2
import math
import os
cos=math.cos
sin =math.sin
pi = math.pi
os.chdir("/Users/shubhamtiwary/file")
length = [3,7]
wid = [1,3]
color = [[255,0,0],[0,0,255]]
r= 0
for i in range(12): # angel
    ang = (pi/12)*i
    for k in range(2): #length
        for l in range(2): #width
            for c in  range(2):#color
                for j in range(1000): #variation
                    
                    p,q = length[k]*abs(math.cos(ang)),length[k]*abs(math.sin(ang))
                    x= np.random.randint(low = length[k]+1,high = 28-length[k],size =1)
                    y= np.random.randint(low = length[k]+1,high = 28- length[k],size =1)
                    x1,y1 = x+ length[k],y + length[k]
                    x2,y2 = x- length[k], y- length[k]  
                    x1_n,y1_n = cos(ang)*(x1-x) -sin(ang)*(y1-y)+ x,sin(ang)*(x1-x) +cos(ang)*(y1-y) + y
                    x2_n,y2_n = cos(ang)*(x2-x) -sin(ang)*(y2-y)+ x,sin(ang)*(x2-x) +cos(ang)*(y2-y) + y
                    img = np.zeros((28,28,3),np.uint8)
                    cv2.line(img,(x1_n,y1_n),(x2_n,y2_n),color[c],wid[l])
                    name = str(i)+"_"+str(k)+"_"+str(l)+"_"+str(c)+str(j)+ ".jpg"
                   # name= str(r) + ".jpg"
                    cv2.imwrite(name,img)
                    r+=1  
                    print(p,q,r,i,x,y)
os.chdir("/Users/shubhamtiwary/file")
frame_array = []
for i in range(12): 
    for k in range(2): 
        for l in range(2): 
            for c in  range(2):
                for j in range(90): 
                    name = str(i)+"_"+str(k)+"_"+str(l)+"_"+str(c)+str(j)+ ".jpg"
                    img = cv2.imread(name)
                    print(img)
                    height, width, layers = img.shape
                    size = (width,height)
                    #print(name)
                    #inserting the frames into an image array
                    frame_array.append(img)
#os.chdir("/Users/shubhamtiwary/file2")
out = cv2.VideoWriter("video_shubham.avi",cv2.VideoWriter_fourcc(*"DIVX"), fps=2,size= 1)
for i in range(len(frame_array)):
    out.write(frame_array[i])
    out.release()
                  

                    