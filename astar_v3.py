import cv2
import numpy as np
from matplotlib import pyplot as plt
import struct
#import serial
import time

#data = serial.Serial('COM10',115200,serial.EIGHTBITS)
#data.write(struct.pack('!B', int(2)))

#Processing the first image
cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('frame', frame)
        cv2.imwrite('img1.png',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
img = cv2.imread('gridmapobs1.png')
img2hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 0, 0])
upper_red = np.array([255, 255, 255])
mask = cv2.inRange(img2hsv, lower_red, upper_red)
res = cv2.bitwise_and(img,img,mask=mask)
#cv2.imwrite('k.png',res)
#res = cv2.resize(res,(500,500))
'''while(1):
    cv2.imshow('image',res)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        cv2.imwrite('config.png',res)
        break'''
cv2.destroyAllWindows()

#A* Algorithm starts
#res = cv2.imread('config.png')

grid_size = 5
rstep = res.shape[0]/grid_size
cstep = res.shape[1]/grid_size
print(rstep,cstep)

grid = []
grid1 = []
grid2 = []
h_val = []
g_val = []
f_val = []
temp = []
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

#Getting the 4 boundaries
corn1 = []
corn2 = []
corn3 = []
corn4 = []

for i in range(0,(grid_size*grid_size),grid_size):
    corn1.append(i)
for i in range(0,grid_size,1):
    corn2.append(i)
for i in range((grid_size-1),(grid_size*grid_size),grid_size):
    corn3.append(i)
for i in range((grid_size*(grid_size-1)),(grid_size*grid_size),1):
    corn4.append(i)

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
print (grid1)
maxi = []
for i in range(0,grid_size*grid_size,1):
    maxi.append(cv2.countNonZero(grid1[i]))
print(maxi)
ind = maxi.index(max(maxi))
#indx = int(ind)/int(grid_size)
#indy = int(ind)%int(grid_size)
indx =0
indy =0
maxi.remove(max(maxi))
ind1 = maxi.index(max(maxi))
ind1x = int(ind1)/int(grid_size)
ind1y = int(ind1)%int(grid_size)
del maxi[:]

print ind,ind1
xe = int(input("Enter x:"))
ye = int(input("Enter y:"))


h = [[0 for x in range(grid_size)]for y in range(grid_size)]
g = [[0 for x in range(grid_size)]for y in range(grid_size)]
f = [[1000 for x in range(grid_size)]for y in range(grid_size)]
ff = [[1000 for x in range(grid_size)]for y in range(grid_size)]
p = [[0 for x in range(grid_size)]for y in range(grid_size)]

for i in range (0,grid_size,1):
 for j in range (0,grid_size,1):
     h[i][j]= 10*(max(abs(i - xe),abs(j -ye))- min(abs(i - xe),abs(j -ye))) + 14 * (min(abs(i - xe),abs(j -ye)))

c_list = [(1,1),(1,2),(1,3),(1,4),(2,1),(3,1),(3,2),(3,3)]
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


