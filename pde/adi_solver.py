from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config_loader import pde_config
from utils.utils import get_logger



logger = get_logger("ADI_SOLVER")

from datetime import datetime
date = datetime.date(datetime.now())

def ADI_solver(id,initial,dx,dy,dt,T,constant,day,thresh,reset = False):
    """
    Input: Initial
    Lx, Ly: 
    """
    Lx, Ly = initial.shape[0], initial.shape[1]
    logger.info("ADI Solver: {}".format(id))
    fig, ax = plt.subplots(figsize=(4, 4))
    frame = 0
    images = []
    list_img = []
    Nx = int(Lx/dx)
    Ny = int(Ly/dy)
    x = np.linspace(0,Lx,Nx) #Create points in direction of x
    y = np.linspace(0,Ly,Ny) #Create points in direction of y
    u = np.ones((Nx,Ny)) #Create Solution Array.
    u_half = np.ones((Nx,Ny)) #Create a solution at t + dt/2
    u_last = np.ones((Nx,Ny)) #Create a solution at t + dt
    path = [pde_config.res_img_path, str(date),id, str(day), "{:.4f}_{:.4f}".format(constant,thresh)]
    path_np = [pde_config.res_npy, id, "{:.4f}_{:.4f}.npy".format(constant,thresh)]


    if os.path.exists(os.path.join(*path_np)) and reset:
        logger.info("LOADED PRE-CALCULATED PARAMS")
        return None, np.load(os.path.join(*path_np), list_img)
    logger.info("RUN ADI ON NEW PARAMS INDEX {}{}".format(constant,thresh))
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path), exist_ok=True)
    C_x = dt/dx**2 #Constant for equation in x part
    C_y = dt/dy**2 #Constant for equation in y part
    #file_path = os.path.join(pde_config.res_img_path,str(date))
    file_path = os.path.join(*path,"final_step.png")
    #Set initial condition: 
    t = 0
    u[:,:] = initial[:,:]
    for t in (range(1, int(T/dt))):     #Loop through t from 0 to T 
        numcount = []
        t_list = []
        #Sweep in y direction
        for j in range(1,Ny-1):
            A = np.zeros(shape = (Nx,Nx))
            b = np.zeros((Nx,1),float)
            i = 0
            A[i,i] = 1 #First in diagonal
            i = Nx-1
            A[i,i] = 1
            i = 0
            b[i] = 1
            for i in range(1,Nx-1):
                A[i,i-1] = -constant*C_x/2
                A[i,i] = 1 + constant*C_x
                A[i,i+1] = -constant*C_x/2
                b[i] = (1-constant*C_y)*u[i,j] + constant*(C_y/2*u[i,j-1]+C_y/2*u[i,j+1])
            
                #Second treat upper boundary
            i = Nx -1
            b[i] = 1
            temp = linalg.solve(A,b)
            for i in range(0,Nx):
                u_half[i,j] = temp[i]
        # Sweep in y direction
        for i in range(1,Nx-1):
            B = np.zeros(shape = (Ny,Ny))
            c = np.zeros((Ny,1),float)
        #First treat left boundary for row i
            j = 0
            B[j,j] = 1 #First in diagonal
            c[j] = 1
            #c[j] = 2*u[i,j] + C_y/2*(u[i,j-1] - 2*u[i,j]+u[i,j+1])
        #Run through inner points for col j
            for j in range(1,Ny-1):
                B[j,j-1] = -constant*C_y/2
                B[j,j] = 1 + constant*C_y
                B[j,j+1] = -constant*C_y/2
                c[j] = (1-constant*C_x)*u_half[i,j] + constant*C_x/2*u_half[i-1,j] +constant*C_x/2*u_half[i+1,j]
            #Second treat right boundary
            j = Ny-1
            B[j,j] = 1
            #c[j] = 0
            c[j] = 1
        #Solve system x = A^(-1) * b
            temp = linalg.solve(B,c)
            #Insert sol into col j
            u_test = u_last
            for j in range(0,Ny):
                u_last[i,j] = temp[j]
        t_list.append(t)
        t+=dt
        u = u_last.copy() 
        show,count = thresh_hold(u,thresh)
        u = show
        numcount.append(count)
        plt.plot(numcount,t_list)
        list_img.append(show)
        im = plt.imshow(show, origin='lower')
        plt.colorbar(shrink=0.5)
        plt.text(0.5, 3, "t = " + str(t))
        #plt.text(5,5, "c1 =" + str(c1))
        plt.savefig(file_path) #Saves each figure as an image
        images.append(imageio.imread(file_path)) #Adds images to list
        plt.clf()

    plt.close("all")
    imageio.mimsave(file_path + ".gif", images, fps=4)
    if not os.path.exists(os.path.join(*path_np)):
        os.makedirs(os.path.join(*path_np[:-1]),exist_ok=True)
    np.save(os.path.join(*path_np), list_img)
    logger.info("SAVED NPY FILE")
    return u,list_img

"""
ADI Solver
"""
def thresh_hold(k,thresh):
    count = 0
    z = np.zeros((k.shape))
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            if(k[i,j] >= thresh):
                z[i,j] = 1
                count +=1
    return z,count

