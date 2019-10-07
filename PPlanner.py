import cv2
import numpy as np
from matplotlib import pyplot as plt
import struct
import serial
import time
import math

boti = 1
while(1):
    if boti == 1:
        data = serial.Serial('COM24',115200,serial.EIGHTBITS)
        r=13
    if boti == 2:
        data = serial.Serial('COM19',115200,serial.EIGHTBITS)
        r=13
        
    #data1 = serial.Serial('COM11',115200,serial.EIGHTBITS)
    #data2 = serial.Serial('COM12',115200,serial.EIGHTBITS)

    #data.write(struct.pack('!B', int(2)))
    data.write(bytes('6'))
    #Processing the first image
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        time.sleep(0.1)
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('frame', frame)
            cv2.imwrite('img1.png',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    img = cv2.imread('img1.png')
    img2hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    '''lower_red = np.array([0, 0, 0])
    upper_red = np.array([255, 255, 255])
    mask = cv2.inRange(img2hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img,img,mask=mask)'''
    #cv2.imwrite('k.png',res)
    res = cv2.resize(img,(480,660))
    while(1):
        cv2.imshow('image',res)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            #cv2.imwrite('config.png',res)
            break
    cv2.destroyAllWindows()

    #A* Algorithm starts
    #res = cv2.imread('config.png')

    grid_size = 6
    rstep = res.shape[0]/grid_size
    cstep = res.shape[1]/grid_size
    rsteph = rstep/2
    csteph = cstep/2
    print(rstep,cstep)
    centre = []
    grid = []
    grid200 =[]
    grid201 = []
    grid300 =[]
    grid301 = []
    grid400 =[]
    grid401 = []
    grid500 =[]
    grid501 = []
    grid700 =[]
    grid701 = []
    grid1 = []
    grid2 = []
    h_val = []
    g_val = []
    f_val = []
    temp = []
    temph= []
    temp_f_val = []
    temp_g_val = []
    o_list = []
    c_list = []
    l_list = []
    p_list= []
    q_list=[]
    hmin =1000
    index = []
    path = []
    #anglea=[]
    moveflag = 0
    turnflag = 0

    if boti == 2:
        c_list.append((xe,ye))
        
    
    for i in range(0,res.shape[0],rstep):
        for j in range (0,res.shape[1],cstep):
            
            temp = res[i:i+rstep,j:j+cstep]
            temp2gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            grid.append(temp)
            grid1.append(temp2gray)

    for i in range(0,(grid_size*grid_size),1):
            plt.subplot(grid_size,grid_size,i+1), plt.imshow(grid[i]), plt.title(i)
            plt.xticks([]), plt.yticks([])
    plt.show()

    for j in range(0,res.shape[1],cstep):
        for i in range (0,res.shape[0],rstep):
            temph = [i+rsteph, j+csteph]
            centre.append(temph)
            
    #TO SWITCH ON BOT1


    #start_point [RED]
    lower_red = np.array([0,120,254])
    upper_red = np.array([8,255,255])
    mask = cv2.inRange(img2hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img,img,mask=mask)
    #kernel = np.ones((5,5),np.uint8)
    #result1 = cv2.erode(res,kernel,iterations = 1)
    kernel = np.ones((5,5),np.uint8)
    result1 = cv2.dilate(res,kernel,iterations = 1)
    res = cv2.medianBlur(result1,9)
    res = cv2.resize(res,(480,660))

    for i in range(0,res.shape[0],rstep):
        for j in range (0,res.shape[1],cstep):
            temp = res[i:i+rstep,j:j+cstep]
            temp2gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            grid200.append(temp)
            grid201.append(temp2gray)
    for i in range(0,(grid_size*grid_size),1):
            plt.subplot(grid_size,grid_size,i+1), plt.imshow(grid200[i]), plt.title(i)
            plt.xticks([]), plt.yticks([])
    plt.show()

    for i in range(0, (grid_size*grid_size),1):
        lmnop = np.count_nonzero(grid200[i])
        if lmnop > 50:
            start = i
            print(start)
            indx = int(start)/int(grid_size)
            indy = int(start)%int(grid_size)
    if boti == 1:
        #end_point [GREEN]
        lower_red = np.array([60,44,60])
        upper_red = np.array([80,255,90])
        mask = cv2.inRange(img2hsv, lower_red, upper_red)
        res = cv2.bitwise_and(img,img,mask=mask)
        kernel = np.ones((3,3),np.uint8)
        result1 = cv2.dilate(res,kernel,iterations = 1)
        res = cv2.medianBlur(result1,9)
        res = cv2.resize(res,(480,660))
    else:
        lower_red = np.array([100,111,215])
        upper_red = np.array([130,125,255])
        mask = cv2.inRange(img2hsv, lower_red, upper_red)
        res = cv2.bitwise_and(img,img,mask=mask)
        kernel = np.ones((8,8),np.uint8)
        result1 = cv2.dilate(res,kernel,iterations = 1)
        res = cv2.medianBlur(result1,9)
        res = cv2.resize(res,(480,660)) 
    
    for i in range(0,res.shape[0],rstep):
        for j in range (0,res.shape[1],cstep):
            temp = res[i:i+rstep,j:j+cstep]
            temp2gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            if boti == 1:
                grid300.append(temp)
                grid301.append(temp2gray)
            if boti == 2:
                grid700.append(temp)
                grid701.append(temp2gray)
                
    for i in range(0,(grid_size*grid_size),1):
            if boti == 1:
                plt.subplot(grid_size,grid_size,i+1), plt.imshow(grid300[i]), plt.title(i)
                plt.xticks([]), plt.yticks([])
            if boti == 2:
                plt.subplot(grid_size,grid_size,i+1), plt.imshow(grid700[i]), plt.title(i)
                plt.xticks([]), plt.yticks([])
    plt.show()
    if boti==1:
        for i in range(0, (grid_size*grid_size),1):
            lmnop = np.count_nonzero(grid300[i])
        
            if lmnop > 1000:
                  end = i
                  print(end)
    else:
        for i in range(0, (grid_size*grid_size),1):
            lmnop = np.count_nonzero(grid700[i])
            if lmnop > 700:
                end = i
                print(end)
                
    xe = int(end)/int(grid_size)
    ye = int(end)%int(grid_size)
#    if boti == 2:
##        #end_point [BLUE]
##        lower_red = np.array([100,111,215])
##        upper_red = np.array([130,125,255])
##        mask = cv2.inRange(img2hsv, lower_red, upper_red)
##        res = cv2.bitwise_and(img,img,mask=mask)
##        kernel = np.ones((3,3),np.uint8)
##        result1 = cv2.dilate(res,kernel,iterations = 1)
##        res = cv2.medianBlur(result1,9)
##        res = cv2.resize(res,(480,660))
#        end=u

    #obstacle [BROWN]
    lower_red = np.array([83,55,110])
    upper_red = np.array([95,90,135])
    mask = cv2.inRange(img2hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img,img,mask=mask)
    kernel = np.ones((5,5),np.uint8)
    result1 = cv2.dilate(res,kernel,iterations = 1)
    res = cv2.medianBlur(result1,7)
    res = cv2.resize(res,(480,660))





    for i in range(0,res.shape[0],rstep):
        for j in range (0,res.shape[1],cstep):
            temp = res[i:i+rstep,j:j+cstep]
            temp2gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            grid400.append(temp)
            grid401.append(temp2gray)
    for i in range(0,(grid_size*grid_size),1):
            plt.subplot(grid_size,grid_size,i+1), plt.imshow(grid400[i]), plt.title(i)
            plt.xticks([]), plt.yticks([])
    plt.show()

    for i in range(0, (grid_size*grid_size),1):
        lmnop = np.count_nonzero(grid400[i])
        if lmnop >1000:
            obstacle = i
            print(obstacle)
            print('****')
            xo = int(obstacle)/int(grid_size)
            yo = int(obstacle)%int(grid_size)
            print(xo,yo)
            c_list.append((xo,yo))


    #start [YELLOW]
    lower_yellow = np.array([24,20,254])
    upper_yellow = np.array([60,100,255])
    mask = cv2.inRange(img2hsv, lower_yellow, upper_yellow)
    res = cv2.bitwise_and(img,img,mask=mask)
    kernel = np.ones((1,1),np.uint8)
    result1 = cv2.erode(res,kernel,iterations = 1)
    kernel = np.ones((7,7),np.uint8)
    result1 = cv2.dilate(res,kernel,iterations = 1)
    res = cv2.medianBlur(result1,7)
    res = cv2.resize(res,(480,660))


    for i in range(0,res.shape[0],rstep):
        for j in range (0,res.shape[1],cstep):
            temp = res[i:i+rstep,j:j+cstep]
            temp2gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            grid500.append(temp)
            grid501.append(temp2gray)
    for i in range(0,(grid_size*grid_size),1):
            plt.subplot(grid_size,grid_size,i+1), plt.imshow(grid500[i]), plt.title(i)
            plt.xticks([]), plt.yticks([])
    plt.show()

    for i in range(0, (grid_size*grid_size),1):
        lmnop = np.count_nonzero(grid500[i])
        if lmnop > 5000:
            startyellow = i
            print(startyellow)
            print('****')
            xyel = int(startyellow)/int(grid_size)
            yyel = int(startyellow)%int(grid_size)

            
            
    h = [[0 for x in range(grid_size)]for y in range(grid_size)]
    g = [[0 for x in range(grid_size)]for y in range(grid_size)]
    f = [[1000 for x in range(grid_size)]for y in range(grid_size)]
    ff = [[1000 for x in range(grid_size)]for y in range(grid_size)]
    p = [[0 for x in range(grid_size)]for y in range(grid_size)]

    for i in range (0,grid_size,1):
     for j in range (0,grid_size,1):
         h[i][j]= 10*(max(abs(i - xe),abs(j -ye))- min(abs(i - xe),abs(j -ye))) + 14 * (min(abs(i - xe),abs(j -ye)))

    o_list = [(indx,indy)]
    i,j = o_list[0]
    g[i][j]=0
    flag=0
    c_list.append ((i,j))

    while(1):
        for t in range (i-1,i+2,1):
            if flag==0:
                #c_list.append ((i,j))
                #o_list.remove ((i,j))
                flag=1
            for y in range (j-1,j+2,1):
                if t<0 or y<0:
                    continue
                elif t> (grid_size-1) or y>(grid_size-1):
                    continue
                elif (t,y) not in c_list:
                    if p[t][y] == 0:
                        p[t][y]= (i,j)
                    o_list.append ((t,y))
                    if(t==i) or (y==j):
                        g[t][y]= 10 + g[i][j]
                    else:
                        g[t][y]= 14 + g[i][j]
                    if f[t][y] > h [t][y] + g [t][y]:
                        f[t][y] = h [t][y] + g [t][y]
                        ff[t][y] = h [t][y] + g [t][y]
                        if p[t][y] != 0:
                            p[t][y] = (i,j)
                           
        for i in range (0,len(c_list),1):
            x,y = c_list[i]
            ff[x][y]=1000
        a=np.argmin(ff)
        i=int(a)/int (grid_size)
        j=int (a) % int (grid_size)
        m=ff[i][j]
        #print(m)
        l_list = [(i,j)]
        for t in range (0,grid_size,1):
            for y in range (0,grid_size,1):
                if t==i and y==j:
                    continue
                if ff[t][y]==m :
                    l_list.append ((t,y))
        #print(l_list)    
        for s in range(0,len(l_list),1):
             t,y = l_list[s]
             #print hmin
             if hmin >h[t][y] :
                hmin = h[t][y]
                i,j=t,y
                flag=0
        c_list.append((i,j))
        #print(h)
        if i==xe and j==ye:
            break


         
    #print(h,g,ff)
    #print(p)

    p_list=[(xe,ye)]
    a=xe
    b=ye

    while(1):
        
        p_list.append(p[a][b])
        a,b=p[a][b]
        if a==indx and b==indy:
            break
    p_list.reverse()
    print(p_list)
    for q in range(0,len(p_list),1):
        a,b = p_list[q]
        path.append((a*grid_size) + b) 

    pix = [[]]
    points = [[]]

    for j in range(((res.shape[0])/(grid_size*2)), (res.shape[0]+1), ((res.shape[0])/(grid_size))):
            for i in range(((res.shape[1])/(grid_size*2)), (res.shape[1]+1) , ((res.shape[1])/(grid_size))):
                    pix.append([i, j])
    del pix[0]
    #print pix

    for i in range(0, len(path), 1):
              points.append(pix[path[i]])
    del points[0]
    #print points

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    img = cv2.polylines(res, [pts], False, (0, 255, 255), 3)

    #del pix[:]
    #del pts[:]

    cv2.imshow('img', img)

    for i in range(0,(grid_size*grid_size),1):
              plt.subplot(grid_size,grid_size,i+1), plt.imshow(grid[i]), plt.title(i)
              plt.xticks([]), plt.yticks([])
    plt.show()
    print path

    # MOVEMENT TO REACH DESTINATION
    current_location = start
    print(current_location)
    cx,cy = p_list[0]
    print(cx,cy)
    pathi = 1
    '''while (current_location != end):
        #calculate current_location at end
        #pathi = pathi + 1
        dummyx,dummyy = p_list[pathi]
        print(dummyx,dummyy)
        if (cx == dummyx):
            if( cy < dummyy):
                print('case7')
                angle = 0
            else:
                print('çase8')
                angle = 180
        elif (cy == dummyy):
            if( cx > dummyx):
                print('case1')
                angle = 90
            else:
                print('çase2')
                angle =  270
        elif ((cx - 1) == dummyx and (cy - 1) == dummyy ):
                print('case4')
                angle = 135
        elif ((cx + 1) == dummyx and (cy + 1) == dummyy ):
                print('case6')
                angle = 315
        elif ((cx + 1) == dummyx and (cy - 1) == dummyy ):
                print('case5')
                angle = 225 # 225
        elif ((cx - 1) == dummyx and (cy + 1) == dummyy ):
                print('case3')
                angle = 45
        #anglea.append(angle)
        current_location = path[pathi]        
        print(current_location)
        cx,cy = p_list[pathi]
        pathi = pathi+1
    print(anglea)'''
    pathi = 1

    while (1):
        
        #time.sleep(0.2)
        #print("***************")
        if turnflag == 0:
            
            #FINDING ORIENTATION RED
            ret, frame = cap.read()
            time.sleep(0.02)
            ret, frame = cap.read()
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            lower_range = np.array ([0,120,254])
            upper_range = np.array([8,255,255])#change for yellow
            mask = cv2.inRange(hsv, lower_range, upper_range)
            result = cv2.bitwise_and(frame,frame,mask=mask)
            #kernel = np.ones((4,4),np.uint8)
            #result1 = cv2.erode(result,kernel,iterations = 1)
            kernel = np.ones((5,5),np.uint8)
            result1 = cv2.dilate(result,kernel,iterations = 1)
            result1 = cv2.medianBlur(result1,9)
            gray = cv2.cvtColor(result1,cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray,1,0.5,1)
            #corners = np.int0(corners)
            if corners is None:
             print("no corner")
            else:    
                 for i in corners:
                     xrpixel,yrpixel = i.ravel()
                     #print(xrpixel,yrpixel)
                     cv2.circle(result1,(x,y),10,255,-1)

            #FINDING ORIENTATION YELLOW
            ret, frame = cap.read()
            time.sleep(0.02)
            ret, frame = cap.read()
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            lower_range = np.array ([20,20,254])
            upper_range = np.array([60,100,255])#change for yellow
            mask = cv2.inRange(hsv, lower_range, upper_range)
            result = cv2.bitwise_and(frame,frame,mask=mask)
            kernel = np.ones((1,1),np.uint8)
            result1 = cv2.erode(result,kernel,iterations = 1)
            kernel = np.ones((7,7),np.uint8)
            result1 = cv2.dilate(result,kernel,iterations = 1)
            result1 = cv2.medianBlur(result1,9)
            gray = cv2.cvtColor(result1,cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray,1,0.5,1)
            #corners = np.int0(corners)
            if corners is None:
                 print("no corner")
            else:    
                 for i in corners:
                     xypixel,yypixel = i.ravel()
                     #print(xypixel,yypixel)
                     cv2.circle(result1,(x,y),10,255,-1)

            #FINDING SLOPE AND ORIENTATION
            if (xrpixel == xypixel):
                xrpixel = xrpixel + 0.1
            slope = (yypixel - yrpixel)/(xypixel - xrpixel)
            #print(slope)
            if (slope > 11 or slope < -11):
                if (yypixel < yrpixel):
                    #print('case1')
                    # check if it will work in bot (-10 to 10) # error carry out =-10
                    anglebot = 90
                else:
                    #print('case2')
                    anglebot = 270
      
            elif (slope > -11 and slope < - 0.08):
                  if (xypixel < xrpixel):
                      #print('case5')
                      anglebot = 180 + (math.atan(abs(slope)))*180/3.14
                  else:
                      #print('case3')
                      anglebot = (math.atan(-1*slope))*180/3.14

            elif (slope > 0.08 and slope < 11):
                  if (xypixel > xrpixel):
                      #print('case6')
                      anglebot = 360+(math.atan(-1*slope))*180/3.14
                  else:
                      #print('case4')
                      anglebot = 180-(math.atan(abs(slope)))*180/3.14
                      #print(anglebot)
            elif (slope > -0.08 or slope < 0.08):
                  if (xypixel < xrpixel):
                      #print('case8')
                      anglebot = 180
                  else:
                      #print('case7')
                      anglebot = 0


            
            #time.sleep(5)
            #print(anglebot)
            #time.sleep(5)
            xc,yc = centre[path[pathi]]

            #FINDING SLOPE AND ORIENTATION
            if (xrpixel == xc):
                xrpixel = xrpixel + 0.1
            slope = (yc - yrpixel)/(xc - xrpixel)
            #print(slope)
            if (slope > 11 or slope < -11):
                if (yc < yrpixel):
                    #print('case1')
                    # check if it will work in bot (-10 to 10) # error carry out =-10
                    anglea = 90
                else:
                    #print('case2')
                    anglea = 270
      
            elif (slope > -11 and slope < - 0.08):
                  if (xc < xrpixel):
                      #print('case5')
                      anglea = 180 + (math.atan(abs(slope)))*180/3.14
                  else:
                      #print('case3')
                      anglea = (math.atan(-1*slope))*180/3.14

            elif (slope > 0.08 and slope < 11):
                  if (xc > xrpixel):
                      #print('case6')
                      anglea = 360+(math.atan(-1*slope))*180/3.14
                  else:
                      #print('case4')
                      anglea = 180-(math.atan(abs(slope)))*180/3.14
                      #print(anglebot)
            elif (slope > -0.08 or slope < 0.08):
                  if (xc < xrpixel):
                      #print('case8')
                      anglea = 180
                  else:
                      #print('case7')
                      anglea = 0
        
            if (anglea - anglebot < -25):
                data.write(bytes(2)) #1 means clockwise
                #time.sleep(0.5)           
            elif(anglea- anglebot > 25):
                data.write(bytes(3)) #2 means anticlock
                #time.sleep(0.5)
            elif (anglea- anglebot < 25) and (anglea - anglebot > - 25) :
                #data.write(bytes(4)) #stop
                #time.sleep(0.5)
                turnflag = 1
                #print("turn done")
           
        if (turnflag == 1):
            #FOR MOVEMENT
            ret, frame = cap.read()
            time.sleep(0.02)
            ret, frame = cap.read()
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            lower_range = np.array ([20,20,254])
            upper_range = np.array([60,100,255])
            mask = cv2.inRange(hsv, lower_range, upper_range)
            result = cv2.bitwise_and(frame,frame,mask=mask)
            #kernel = np.ones((4,4),np.uint8)
            #result1 = cv2.erode(result,kernel,iterations = 1)
            kernel = np.ones((5,5),np.uint8)
            result1 = cv2.dilate(result,kernel,iterations = 1)
            result1 = cv2.medianBlur(result1,9)
            gray = cv2.cvtColor(result1,cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray,1,0.5,1)
            #corners = np.int0(corners)
            if corners is None:
                 print("no corner")
            else:    
                 for i in corners:
                     xrpixel,yrpixel = i.ravel()
                     #print(xrpixel,yrpixel)
                     cv2.circle(result1,(x,y),10,255,-1)
            xc,yc = centre[path[pathi]]
            rc = ((xrpixel - xc)**2 + (yrpixel - yc)**2)**0.5
            #print(rc)
            if  rc > r:
                data.write(bytes(1)) #forward
                #print("Forward")
                turnflag = 0
                time.sleep(0.05)
            elif r > rc:
                turnflag = 0
                #data.write(bytes(5))
                if path[pathi] == end:
                    u=path[pathi]
                    data.write(bytes(7))
                    boti = boti+1
                    time.sleep(0.3)
                    break
                
                if path[pathi+1] == end:
                    #if boti==2:
                        r=45
                pathi = pathi + 1
    if boti==3:
        break
cap.release()    
