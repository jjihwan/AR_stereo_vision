import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


oldx = oldy = -1 # 좌표 기본값 설정

def on_mouse(event, x, y, flags, data):
    # event는 마우스 동작 상수값, 클릭, 이동 등등
    # x, y는 내가 띄운 창을 기준으로 좌측 상단점이 0,0이 됌
    # flags는 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미, Shif+마우스 등 설정가능
    # param은 영상이룻도 있도 전달하고 싶은 데이타, 안쓰더라도 넣어줘야함
    if event == cv.EVENT_LBUTTONDOWN: # 왼쪽이 눌러지면 실행
        data.append([x,y])
        print('EVENT_LBUTTONDOWN: %d, %d' % (x, y)) # 좌표 출력

def clickImg(img):
    data = []
    # 윈도우 창
    cv.namedWindow('image')

    # 마우스 입력, namedWIndow or imshow가 실행되어 창이 떠있는 상태에서만 사용가능
    # 마우스 이벤트가 발생하면 on_mouse 함수 실행
    cv.imshow('image', img)
    cv.setMouseCallback('image', on_mouse, data)

    print(data)
    # 영상 출력
    cv.waitKey()

    cv.destroyAllWindows()

    # 유저가 마우스로 찍은 점을 float로 바꿔야 함.
    clicked2D = np.array(data, dtype=float)
    return clicked2D

def makeCube(img, plane, K, X3D):
    print("plot.py : object select...")
    
    obj2Dimg = np.array(clickImg(img))
    c = np.array([K[0][2],K[1][2]])
    f = np.array([K[0][0],K[1][1]])

    obj2Dnorm = (obj2Dimg-c)/f
    k = -plane[3]/(plane[0]*obj2Dnorm[:,0]+plane[1]*obj2Dnorm[:,1]+plane[2]) # constant multiple to project to plane
    k = np.expand_dims(k, axis=0)
    obj3Dxy = k.T*obj2Dnorm # (-k*a,-k*b,z) in the plane is the point that projects to image point (0,0) 
    obj3Dz = -(plane[0]*obj3Dxy[:,0]+plane[1]*obj3Dxy[:,1]+plane[3])/plane[2]
    obj3D = np.concatenate((obj3Dxy, np.expand_dims(obj3Dz,1)),1)
    print(obj3D)
    obj3D = np.concatenate((obj3D, obj3D-plane[0:3]*5), 0)
    
    meshx_plane, meshy_plane = np.meshgrid(range(-30,20),range(-20,20))
    meshz_plane = -(plane[0]*meshx_plane+plane[1]*meshy_plane+plane[3])/plane[2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("XYZ coordinates relative to second camera")
    ax.set_xlabel("x")
    ax.set_xlim(-20, 20)
    ax.set_ylabel("y")
    ax.set_ylim(-20, 20)
    ax.set_zlabel("z")
    ax.set_zlim(0, 50)
    ax.plot_surface(meshx_plane,meshy_plane,meshz_plane,alpha=0.2)
    ax.scatter(obj3D[:,0],obj3D[:,1],obj3D[:,2],marker='.', s=10)
    ax.scatter(X3D[:,0], X3D[:,1], X3D[:,2], marker='o', s=15)
    plt.show()

    return obj3D
