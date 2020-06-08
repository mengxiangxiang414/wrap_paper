import cv2
import numpy as np

#A4纸的长宽比例
ratio = 210/297

#读取原图im1，获取轮廓im2
im1 = cv2.imread('5.jpg')
im2 = np.zeros([np.shape(im1)[0],np.shape(im1)[1]],dtype=np.uint8)
edges = cv2.Canny(im1,100,200)
image, contours, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#筛选出最大轮廓
cnt = []
for i in contours:
    if len(i)>len(cnt):
        cnt = i

#根据轮廓，来生成近似轮廓（四边形）
epsilon = 0.08*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

#如果近似轮廓的表述结构为四个点（即四边形的四个顶点），则计算最长的边长
if len(approx)==4:
    len_1 = (approx[0,0,0]-approx[1,0,0])**2+(approx[0,0,1]-approx[1,0,1])**2
    len_2 = (approx[1, 0, 0] - approx[2, 0, 0])**2 + (approx[1, 0, 1] - approx[2, 0, 1])**2
    len_3 = (approx[2, 0, 0] - approx[3, 0, 0])**2 + (approx[2, 0, 1] - approx[3, 0, 1])**2
    len_4 = (approx[3, 0, 0] - approx[1, 0, 0])**2 + (approx[3, 0, 1] - approx[1, 0, 1])**2

    len_max = np.sqrt(max([len_1,len_2,len_3,len_4]))
    y_max = np.uint16(len_max)
    x_max = np.uint16(y_max*ratio)

    #透视变换的四个参考点
    base_points = np.array([[0,0],
                            [0,y_max],
                            [x_max,y_max],
                            [x_max,0]])

    #透视变换的四个待变换的点,由于检测到的四个点的顺序可能是混乱的，需要整理它们的顺序，使之分别与四个参考点对应
    dis = [(approx[0,0,0])**2+(approx[0,0,1])**2,
              (approx[1,0,0])**2+(approx[1,0,1])**2,
              (approx[2,0,0])**2+(approx[2,0,1])**2,
              (approx[3,0,0])**2+(approx[3,0,1])**2]
    p0 = dis.index(min(dis));
    change_points = []
    change_points[0 :4-p0] = approx[p0:]
    change_points[4-p0:] = approx[0 :p0]

    pts1 = np.float32(change_points)
    pts2 = np.float32(base_points)

    #获取透视变换矩阵M，并根据M讲原图im1进行变换
    M=cv2.getPerspectiveTransform(pts1,pts2)
    dst=cv2.warpPerspective(im1,M,(x_max,y_max))

    #锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst2 = cv2.filter2D(dst, -1, kernel=kernel)

    #输出与展示
    cv2.imwrite("d:/wrap/21.jpg",dst)
    cv2.imwrite("d:/wrap/22.jpg",dst2)
    cv2.imshow('image',dst)
    cv2.imshow('image2',dst2)
    cv2.waitKey(0)