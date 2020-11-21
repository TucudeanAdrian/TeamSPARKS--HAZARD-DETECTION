
import cv2
import numpy as np
from numpy import matrix
import time
import math
from edge import canny_edge_detector,region_of_interest,display_lines,average_slope_intercept

def nothing(x):
    pass


sensivity=40                                               


capture=cv2.VideoCapture("dz_Trim.mp4") ###pentru video  ((((Cu resolutia 640 pe 480)))
ret,frame=capture.read()
height = frame.shape[0]
width = frame.shape[1]
frame= frame[0:height-200, 0:width]
height = frame.shape[0]
width = frame.shape[1]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out123.avi', fourcc, 20.0, (1280,694))


time.sleep(0.1)
#capture.set(3,row)
#capture.set(4,col)
y_form=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
x1=320
y1=350
a=0
b=0
a=[0,50,0]
intoarcere=0

print("3")
print("2")
print("1")

print("START !!!")

direc=0
agX=800
agY=100
x_cil=0
y_cil=0
danger=0
def FSD(src):
    
  
    
    danger=0
    font = cv2.FONT_HERSHEY_SIMPLEX
  
    while True:
        #kernel=np.ones((3,3),np.uint8)
        kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        c=10
        c1=10
        c2=10
        c3=10
        c4=10
        c5=10
        c6=10
        c7=10
        c8=10
        c9=10
        c10=10
        c11=10
        c12=10
        c13=10   # that is a pro gamer move
        c14=10
        c15=10
        c16=10
        c17=10
        c18=10
        c19=10
        c20=10
        c21=10
        c22=10
        c23=10
        c24=10
        c25=10
        c26=10
        c27=10
        c28=10
        c29=10
        c30=10
        c31=10
        c32=10
        c33=10
        c34=10
        c35=10
        c36=10
        c37=10
        c38=10
        c39=10
        arg=0
        contor=0
        x_form=[c,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35,c36,c37]
        Gx=0
        Gy=0
        Gx1=0
        Gy1=0
        GxR=0
        GyR=0
        row=width
        col=height
        R=0
        ret, frame = capture.read()
        frame= frame[0:height, 0:width]
        x_cil=(int)(width/2)
        y_cil=(int)(height/2)
        l=0
        
        #-----------------------------------------------------------------------------------------------------------------------------------
        window=frame
        #lower_green = np.array([30,60,60])
        #upper_green = np.array([120,255,255])
        
        
        
        #cv2.line(window,(0, int(height/2)),(width,int(height/2) ),(0,255,0),1)

#Our frame, the HSV image, is thresholded among upper and lower pixel ranges to get only green colors
       
        
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #mask = cv2.inRange(hsv, lower_green, upper_green)
        edge=canny_edge_detector(frame)
        edge=cv2.Canny(frame,50,150)
        
        cropped_image = region_of_interest(edge,width,height)
        cr=cv2.dilate(cropped_image,kernel2,1)
        #------------------------------------------------------------------------------------------------
        contours, hie = cv2.findContours(cr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try:
            hie = hie[0]
            for component in zip(contours,hie):
        
                currentContour = component[0]
                currentHierarchy = component[1]
                area = cv2.contourArea(currentContour)
                M = cv2.moments(currentContour)
                if area>200 and currentHierarchy[3] < 0:
                    #cv2.drawContours(frame,currentContour,-1,(0,255,255),1)
            #currentContour==a
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    mask_color = (0,255,0)
                    frame_copy = window.copy()
                    cv2.fillPoly(frame_copy, [currentContour], mask_color)
                    opacity = 0.2
                    cv2.addWeighted(frame_copy,opacity,window,1-opacity,0,window)
                    #cv2.putText(window,"!",(cx,cy),font,1, (0,0,255), 2, cv2.LINE_AA)
            #for c1 in [c]:
            #print(hie)
                    #cv2.fillPoly(frame,[currentContour],(0,240,0))
        except:
            pass
        #-----------------------------------------------------------------------------------------------
        cropped_image= cropped_image[int(2/3*height):height, 0:width]
        
        
        #lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 30, maxLineGap=200)
        #averaged_lines = average_slope_intercept(window,cropped_image)
        #line_image = display_lines(window, lines)
        #combo_image = cv2.addWeighted(window, 0.8, line_image, 1, 1) 
        contorG=0
        contorR=0
        okRin=0
        okRout=0
        
        erosion=cv2.dilate(edge,kernel2,1)
        pic=np.array(erosion)
        for i in range(38):
           y_form[i]=int(width/2-152)+8*i
        for i in range(40,height-40):
            for j in range(38):
            	if pic[i][y_form[j]]: #and c1>i:
                	    if x_form[j]<i:
                      		x_form[j]=i
        cv2.line(window,(10,0),(10,height-10),(255,255,255),1)
        cv2.line(window,(0,height-10),(width-10,height-10),(255,255,255),1)
        for i in range(0,height,(int)(height/23)):
            if(i%4==0):
                cv2.line(window,(10,i),(40,i),(255,255,255),2)
            else:
                cv2.line(window,(10,i),(20,i),(255,255,255),2)
        cv2.circle(window,((int)(width/2),height),40,(255,255,255),2)
        for j in range(38):
            if x_form[j]>(4*height/5+20):
                
            	cv2.circle(window,(y_form[j],x_form[j]),2,(0,0,255),1)
            	cv2.line(window,(y_form[j],x_form[j]),(y_form[j],height),(0,0,255),1)#rosu
            	contorR=contorR+1
            	
            	if j<37 and x_form[j]>(4*height/5) and x_form[j+1]>(4*height/5):
                        #R=R+1
                        #GxR=GxR+y_form[j]
            		GyR=GyR+x_form[j]
            		#cv2.line(window,(y_form[j+1],x_form[j+1]-30),(y_form[j],x_form[j]-30),(0,0,255),2)
            		GxR=GxR+y_form[j]
            		R=R+1
            
            		
                        
            		
            elif x_form[j]>(3*height/5+50) and x_form[j]<=(4*height/5+20):
            	cv2.circle(window,(y_form[j],x_form[j]),2,(0,255,255),1)
            	Gx=Gx+y_form[j]
            	Gy=Gy+x_form[j]
            	contor=contor+1
            	cv2.line(window,(y_form[j],x_form[j]),(y_form[j],height),(0,255,255),1)
            	#cv2.line(window,(y_form[j],x_form[j]),(y_form[j],x_form[j]-30),(0,255,255),1)
            	#if j<37 and x_form[j]>(3*height/5+50) and x_form[j+1]>(3*height/5+50) and x_form[j]<=(2*height/3+100) and x_form[j+1]<=(2*height/3+100):
            		#cv2.line(window,(y_form[j+1],x_form[j+1]-30),(y_form[j],x_form[j]-30),(0,255,255),1)
                
            else:
            	contor=contor+1
            	contorG=contorG+1
            	
            	Gx1=Gx1+y_form[j]
            	Gy1=Gy1+x_form[j]
            	cv2.circle(window,(y_form[j],x_form[j]),2,(0,255,0),1)
            	#cv2.line(window,(y_form[j],x_form[j]),(y_form[j],x_form[j]-30),(0,255,0),1)
            	cv2.line(window,(y_form[j],x_form[j]),(y_form[j],height),(0,255,0),1)
            	x_cil=x_cil+y_form[j]
            	y_cil=y_cil+x_form[j]
            	#if j<37 and x_form[j]<(3*height/5+50) and x_form[j+1]<(3*height/5+50):
            		#cv2.line(window,(y_form[j+1],x_form[j+1]-30),(y_form[j],x_form[j]-30),(0,255,0),1)
            		
            		
        #Directie(erosion,window,width,height)
  
            
        if contorG==0:
            try:
                Gx=int(Gx/contor)
                Gy=int(Gy/contor)
                A=np.array([Gx,Gy])
            #dist=math.pow((width/2-Gx)*(width/2-Gx)+(height*Gy)*(height*Gy),1/2)
                O=np.array([width,height])
                B=np.array([int(width/2),height])
                AB = B - A
                OB = B - O
                #cosine_angle = np.dot(AB, OB) / (np.linalg.norm(AB) * np.linalg.norm(OB))
                #angle = int(np.degrees(np.arccos(cosine_angle)))
                #cv2.arrowedLine(window,(int(width/2),height),(Gx,Gy),(0,0,0),3)
                #cv2.arrowedLine(window,(int(width/2),height),(Gx,Gy),(255,255,255),2)
                #cv2.putText(window,"Angle "+str(np.around(angle+15,2)),(50,50),font, 1,(0,0,255),2,cv2.LINE_AA)
            except:
                pass
        else:
            Gx1=int(Gx1/contorG)
            Gy1=int(Gy1/contorG)
            A=np.array([Gx1,Gx1])
            #cv2.arrowedLine(window,(int(width/2),height),(Gx1,Gy1),(0,0,0),3)
            O=np.array([width,height])
            B=np.array([int(width/2),height])
            AB = B - A
            OB = B - O
            #cosine_angle = np.dot(AB, OB) / (np.linalg.norm(AB) * np.linalg.norm(OB))
            #angle = int(np.degrees(np.arccos(cosine_angle)))
            #cv2.arrowedLine(window,(int(width/2),height),(Gx1,Gy1),(255,255,255),2)
        cv2.rectangle(window, (40,30), (200,180), (0,0,0), -1)
        cv2.putText(window,"Warnings: "+str(contorR),(50,50),font, 0.5,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(window,"Unknow:"+str(contor),(int(50),75),font, 0.5,(0,255,255),1,cv2.LINE_AA)
        cv2.putText(window,"Safe: "+str(contorG),(50,100),font, 0.5,(0,255,0),1,cv2.LINE_AA)
        
        #cv2.putText(window,"CNT DANGERS: "+str(danger),(50,125),font, 0.5,(0,255,0),1,cv2.LINE_AA)
    
        if R>5:
             #cv2.rectangle(window, (int(GxR/R)-5*R,int(GyR/R)+2*R), (int(GxR/R)+5*R,int(GyR/R)-2*R), (0,200,255), 2)
             #cv2.rectangle(window, (int(GxR/R-20),int(GyR/R)), (int(GxR/R)+20,int(GyR/R)-20), (0,0,0), -1)
             #cv2.putText(window,"!!!",(int(GxR/R),int(GyR/R)),font, R*0.05,(0,0,0),2,cv2.LINE_AA)
             
             #cv2.putText(window,"!!!",(int(GxR/R),int(GyR/R)),font, R*0.05,(0,0,255),1,cv2.LINE_AA)
             if int(GyR/R)>height-100 and int(GyR/R)<height-90:
        # :
                 danger=danger+1
             #if contorG>0:
             #    cv2.arrowedLine(window,(int(width/2),height),(Gx1,Gy1),(0,0,0),3)
             
        if contorR>20:
            cv2.putText(window,"Reduce Speed",(int(width/2-50),100),font, 1,(0,0,255),2,cv2.LINE_AA)
            
            
            #cv2.circle(window,(int(GxR/R),int(GyR/R)),20,(255,255,255),4)
       
            
                
        #cv2.circle(window,(int(width/2),height),140,(255,255,255),1)
        
        
       
        
        cv2.line(window,(300, int(height-100)),(width-500, int(height-100)),(0,0,0),1)
        cv2.line(window,(150, height), (300, int(height-100)),(0,0,0),1)
        cv2.line(window,(width-500, int(height-100)),(width-350, height),(0,0,0),1)
        #cv2.imshow('video', window)
        cv2.line(window,(300, height),(300, 0),(255,255,255),2)
        cv2.line(window,(int(width-300), height),(int(width-300), 0),(255,255,255),2)
        #vis = np.concatenate((window, cropped_image), axis=1)
        depth_image_3d = np.dstack((cropped_image,cropped_image,cropped_image))
        vis = np.concatenate((window, depth_image_3d), axis=0)
        h = vis.shape[0]
        w = vis.shape[1]
        cv2.putText(vis,"Width"+str(w),(int(50),150),font, 0.5,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(vis,"Height"+str(h),(int(50),175),font, 0.5,(0,0,255),1,cv2.LINE_AA)
        cv2.imshow("crop",vis)
        
        
        #print(h)
        #print(w)
        
        
        out.write(vis)
        
        
        #cv2.imshow('Final Mask',erosion)                 ########################              masca
    
        if cv2.waitKey(1) == 27:
            break
    
    capture.release()
    cv2.destroyAllWindows()

FSD(direc)
#fisier.write(str(time.time()-s))
