# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import random

import csv

import matplotlib.pyplot as plt
import OP_joint_data as opdata
import numpy as np
import json
import math
import cv2
import os
from scipy.signal import find_peaks

from scipy.interpolate import interp1d


# def interpolate(function):
#     posiciones=np.array(range(numframes))
#     noerror=np.where(function > 0)
#     interp=interp1d(posiciones[noerror],function[noerror], kind='linear',bounds_error=False,fill_value="extrapolate")
#     f=interp(posiciones)
#     final=np.where(f >= 0, f, 0)
#     return final

def interpolate(function):
    posiciones=np.array(range(len(function)))
    noerror=np.where(function != 0)
    interp=interp1d(posiciones[noerror],function[noerror], kind='linear',bounds_error=False,fill_value="extrapolate")
    f=interp(posiciones)
    final=np.where(f != 0, f, 0)
    return final

def smoothingkeypoints(frame_count,poses):
    smoothkeypoints=poses
    for i in range(25):
        # if frame_count<100:
        #     wind_length=20
        # else:
        #     wind_length=40
        wind_length=20
        ceros=np.count_nonzero(poses[:,i,0]==0)
        if (frame_count-2)>ceros:
            posex=interpolate(poses[:,i,0])
            posey=interpolate(poses[:,i,1])
            newposesx=smooth(posex,wind_length,'hanning')
            newposesy=smooth(posey,wind_length,'hanning')
            # newposesx=smooth(posex,80,2)
            # newposesy=smooth(posey,80,2)
            smoothkeypoints[:,i,0]=newposesx
            smoothkeypoints[:,i,1]=newposesy
    return smoothkeypoints


def smooth(x,window_len,window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError( "Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']:
        raise ValueError( "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    y=y[(int(window_len/2-1)):-(int(window_len/2))]
    return y


def euclidean_distance(vec1, vec2):
    # Se comprueba que los vectores tengan la misma longitud
    if len(vec1) != len(vec2):
      raise ValueError("Los vectores deben tener la misma longitud")
    # Inicialización de la distancia euclidiana a 0
    distance = 0
    # Iteramos sobre los elementos de los vectores y acumulamos
    # la distancia euclidiana en la variable distance
    for i in range(len(vec1)):
      distance += (vec2[i] - vec1[i])**2
    # La distancia euclidiana es la raiz de la variable distance
    return math.sqrt(distance)


def vector_euclidean_distance(vec1, vec2):
    # Comprobamos que los vectores tengan la misma longitud
    if len(vec1) != len(vec2):
      raise ValueError("Los vectores deben tener la misma longitud")
    # Inicializamos el vector de distancias a una lista vacía
    distances = []
    # Iteramos sobre los elementos de los vectores y añadimos
    # a la lista distances la distancia euclidiana entre ellos
    for i in range(len(vec1)):
      distance = euclidean_distance(vec1[i], vec2[i])
      distances.append(distance)
    # Devolvemos el vector de distancias
    return np.array(distances)


def dist_ciclo(DCfram):
    dist_horizontal= abs(DCfram[:,11,0]-DCfram[:,14,0])#distancia horizontal
    dist_euclidiana= vector_euclidean_distance(DCfram[:,11], DCfram[:,14])
    return [dist_horizontal, dist_euclidiana]


def encontrar_indices_maximos(vector,d):
    vector=list(vector)
    average=sum(vector)/(len(vector))*0.75

    a=vector[:d]
    b=vector[-d:]
    
    indices_maximos = []
    for i in range(d, len(vector)-d):  #vector[:-d] es el vector sin los ultimod d elementos
                                            #el rango se salta los primeros y ultimos d elementos
        ver_max=0                             
        if vector[i] > vector[i-1] and vector[i] > vector[i+1]:
            for j in range(2,d+1):    
                if vector[i] > vector[i-j] and vector[i] > vector[i+j]:
                    ver_max += 1
            if ver_max==d-1:
                indices_maximos.append(i)
                
    if max(a)>=average:
        indices_maximos.insert(a.index(max(a)),0)
    if max(b)>=average:
        indices_maximos.append(len(vector)+b.index(max(b))-d)
    print(indices_maximos)
    return indices_maximos


def sum_consecutive(vector):
    # Inicializamos el vector de resultado con el primer elemento del vector de entrada
    result = [] 
    # Iteramos sobre el vector de entrada a partir del segundo elemento
    for i in range(1, len(vector)):
      # Calculamos la suma del elemento actual y el anterior
      s = vector[i] + vector[i - 1]
      # Añadimos la suma al vector de resultado
      result.append(s) 
    return result


def get_height(GHfram):
    arr1=GHfram[:,11,1]
    arr2=GHfram[:,14,1]
    arr3=GHfram[:,1,1]
    min_array = []
    
    for a, b in zip(arr1, arr2):
        min_array.append(max(a, b))
    arr4=abs(arr3-min_array)
    
    altura=sum(arr4)/len(arr4)
    altura=altura*dist_real/image_width
    
    return altura


def calcular_vet(CVfram, waist_mov, dr, fps):
    diff_h = (CVfram[:,11,0]-CVfram[:,14,0])
    diff_h_abs = abs(diff_h)
    peaks,_ = find_peaks(diff_h_abs, height=30, distance=20)
    zeros,_ = find_peaks(-diff_h_abs, height=-20, distance=20)
    amp_lat_max = diff_h[peaks]
    amp_max_R = amp_lat_max[amp_lat_max > 0]
    amp_max_L = -amp_lat_max[amp_lat_max < 0]
    
    if len(amp_max_L)==0:
        amp_max_L = amp_max_R
    if len(amp_max_R)==0:
        amp_max_R = amp_max_L
    
    amp_max_R = sum(amp_max_R)/len(amp_max_R)*(dr/image_width)
    amp_max_L = sum(amp_max_L)/len(amp_max_L)*(dr/image_width)
    
    amp_max_R, amp_max_L = amp_max_L, amp_max_R 
    
    marks = np.sort(np.concatenate((peaks, zeros)))
    medians = np.round((marks[:-1] + marks[1:]) / 2).astype(int)
    
    ntR = sum(diff_h[medians] > 0)
    ntL = sum(diff_h[medians] < 0)
    
    tR = sum(diff_h[marks[0]:marks[-1]] > 0)
    tL = sum(diff_h[marks[0]:marks[-1]] < 0)
    
    if ntL ==0:
        ntL = ntR
        tL = tR
    if ntR ==0:
        ntR = ntL
        tR = tL
    
    tR = ((tR/ntR)/fps)*2
    tL = ((tL/ntL)/fps)*2
    
    v=-(np.mean(np.diff(waist_mov))*dr*fps)/(image_width)
    dR = v*tR
    dL = v*tL
    
    dRight_shoulder_knee=vector_euclidean_distance(CVfram[:,2], CVfram[:,10])
    dLeft_shoulder_knee=vector_euclidean_distance(CVfram[:,5], CVfram[:,13])
    max_dRight_sk=max(dRight_shoulder_knee)*(dr/image_width)
    min_dRight_sk=min(dRight_shoulder_knee)*(dr/image_width)
    max_dLeft_sk=max(dLeft_shoulder_knee)*(dr/image_width)
    min_dLeft_sk=min(dLeft_shoulder_knee)*(dr/image_width)
    height=get_height(CVfram)
    
    return [amp_max_R, amp_max_L, tR, tL, dR, dL, v, max_dRight_sk, min_dRight_sk, max_dLeft_sk, min_dLeft_sk, height]
    

def calcular_vet2(CVfram, waist_mov, dr):     #Calcular variables espacio/temporales
    amp_lat_max=[]
    amp_peaks=[]
    amp_valleys=[]
    
    dist_h=dist_ciclo(CVfram)[0] #[0] horizontal [1] euclidiana
    
    amp_lat_max = max(dist_h)*(dr/image_width)
    amp_peaks, _ = find_peaks(dist_h, height=30, distance=20, prominence=27)
    amp_valleys, _ = find_peaks(-dist_h, height=-20, distance=20)
    
    if len(amp_peaks)==1 and len(amp_valleys)==1:
        t_half=2*abs(amp_peaks-amp_valleys)
    else:
        t1=np.diff(amp_peaks)
        t2=np.diff(amp_valleys)
        t_half=np.concatenate((t1,t2))
    t_half=sum(t_half)/(len(t_half)*60)
    
    if len(amp_peaks)>=len(amp_valleys):
        cycle_marks=amp_peaks
    else: 
        cycle_marks=amp_valleys
    
    if len(cycle_marks)%2==0 and len(cycle_marks)>3:
        cycle_marks= np.delete(cycle_marks, -1)
    t_total1=sum(np.diff(cycle_marks))*60
    t_total2=2*t_half
    v=-(np.mean(np.diff(waist_mov))*dr*60)/(image_width)
    d_half=v*t_half
    d_total=v*t_total2
    dRight_shoulder_knee=vector_euclidean_distance(CVfram[:,2], CVfram[:,10])
    dLeft_shoulder_knee=vector_euclidean_distance(CVfram[:,5], CVfram[:,13])
    max_dRight_sk=max(dRight_shoulder_knee)*(dr/image_width)
    min_dRight_sk=min(dRight_shoulder_knee)*(dr/image_width)
    max_dLeft_sk=max(dLeft_shoulder_knee)*(dr/image_width)
    min_dLeft_sk=min(dLeft_shoulder_knee)*(dr/image_width)
    height=get_height(CVfram)
    
    return [amp_lat_max, t_half, t_total2, d_half, d_total, v, max_dRight_sk, min_dRight_sk, max_dLeft_sk, min_dLeft_sk, height]
    

    
def getAngle(a, b, c): # angulo de C a A en sentido horario
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def calcular_angulos(frames_ang):
    Hip=[]      #Tronco o cadera central    0
    Rhip=[]     #Cadera derecha             1
    Lhip=[]     #Cadera izquierda           2
    Rknee=[]    #Rodilla derecha            3 
    Lknee=[]    #Rodilla izquierda          4
    Rankle=[]   #Tobillo derecho            5
    Lankle=[]   #Tobillo izquierdo          6
    Rshank=[]   #Canilla derecho            7
    Lshank=[]   #Canilla izquirda           8
    Legs=[]     #Ángulo entre las piernas   9
    Relbow=[]   #Codo derecho               10
    Lelbow=[]   #Codo izquierdo             11
    Rshoulder=[]#Hombro derecho             12
    Lshoulder=[]#Hombro izquierdo           13
    
    
    for i in range(len(frames_ang)):
        angle1=getAngle(frames_ang[i,8]-[100,0], frames_ang[i,8], frames_ang[i,1])   #Tronco o cadera central C
        
        angle2=getAngle(frames_ang[i,10], frames_ang[i,9], frames_ang[i,2])         #Cadera lateral D
        angle3=getAngle(frames_ang[i,13], frames_ang[i,12], frames_ang[i,5])
        
        angle4=getAngle(frames_ang[i,9], frames_ang[i,10], frames_ang[i,11])        #Rodilla A
        angle5=getAngle(frames_ang[i,12], frames_ang[i,13], frames_ang[i,14])
        
        angle6=getAngle(frames_ang[i,22], frames_ang[i,11], frames_ang[i,10])       #Tobillo E
        angle7=getAngle(frames_ang[i,19], frames_ang[i,14], frames_ang[i,13])
        
        angle8=getAngle(frames_ang[i,11]-[100,0], frames_ang[i,11], frames_ang[i,10])   #Shank B
        angle9=getAngle(frames_ang[i,14]-[100,0], frames_ang[i,14], frames_ang[i,13])
        
        if angle3 >180:
            angle3=180
        if angle4 >180:
            angle4=180
        
        
        if angle8 >= 180:           #Los puntos entre los que se calcula el ángulo se entrecruzan
            angle8=angle8-360
        if angle9 >= 180:
            angle9=angle9-360
            
        angle10=getAngle(frames_ang[i,10], frames_ang[i,8], frames_ang[i,13])       #Ángulo entre las piernas F
        
        if angle10 >= 180:           #Este angulo es absoluto 
            angle10=360-angle10
        
        angle11=getAngle(frames_ang[i,4], frames_ang[i,3], frames_ang[i,2])         #Codo G
        angle12=getAngle(frames_ang[i,7], frames_ang[i,6], frames_ang[i,5]) 
        
        angle13=getAngle(frames_ang[i,9], frames_ang[i,2], frames_ang[i,3])         #Hombro H
        angle14=getAngle(frames_ang[i,12], frames_ang[i,5], frames_ang[i,6])  
        
        if angle13 >= 180:          #Los puntos entre los que se calcula el ángulo se entrecruzan
            angle13=angle13-360
        if angle14 >= 180:
            angle14=angle14-360
        
        Hip.append(angle1)
        Rhip.append(angle2)
        Lhip.append(angle3)
        Rknee.append(angle4)
        Lknee.append(angle5)
        Rankle.append(angle6)
        Lankle.append(angle7)
        Rshank.append(angle8)
        Lshank.append(angle9)
        Legs.append(angle10)     
        Relbow.append(angle11)   
        Lelbow.append(angle12)   
        Rshoulder.append(angle13)
        Lshoulder.append(angle14)
        
    datos = [Hip, Rhip, Lhip, Rknee, Lknee, Rankle, Lankle, Rshank, Lshank, Legs, Relbow, Lelbow, Rshoulder, Lshoulder]

    return datos
    

def find_discontinuities(array):
    """
    Encuentra las posiciones de discontinuidades en un array.
    """
    diffs = np.diff(array)
    indices = np.where(diffs > 1000)[0]
    nocero=np.where(array[indices]>0)[0]
    indices=indices[nocero]
    if len(indices)==0:
        indices=[0]
    return indices[0]

def find_separation(FSframes):
    discontinuities=[]
    for i in range(25):
        vector=FSframes[:,i,0]
        discontinuities.append(find_discontinuities(vector))
    video_separation=max(set(discontinuities), key = discontinuities.count)
    return video_separation


def find_ratio(FRframes):
    frames1=FRframes[:video_separation+1]
    frames2=FRframes[video_separation+1:]

    spine1=[]
    spine2=[]
    
    for i in range(len(frames1)):
        if frames1[i,1,0]!=0 and frames1[i,8,0]!=0:
            spine1.append(euclidean_distance(frames1[i,1],frames1[i,8]))

    for i in range(len(frames2)):
        if frames2[i,1,0]!=0 and frames2[i,8,0]!=0:
            spine2.append(euclidean_distance(frames2[i,1],frames2[i,8]))
    
    prom1=sum(spine1)/len(spine1)
    prom2=sum(spine2)/len(spine2)

    ratio=prom1/prom2
    
    return ratio


def move_frames(MFframes):
    Centered_frames = np.zeros_like(MFframes)
    for i in range(len(MFframes)):
        offset=(170-MFframes[i][8][0],125-MFframes[i][8][1])
        for j in range(25):
            if MFframes[i][j][0]==0 and MFframes[i][j][1]==0:
                Centered_frames[i][j][0]=0
                Centered_frames[i][j][1]=0
            else:
                Centered_frames[i][j][0]=MFframes[i][j][0]+offset[0]
                Centered_frames[i][j][1]=MFframes[i][j][1]+offset[1]
    return Centered_frames

    
def get_waist_mov(GWMfram):
    waist_mov=[]
    vds=video_separation
    waist_mov=GWMfram[:,8,0]
    value=waist_mov[vds+1]-waist_mov[vds]
    waist_mov[vds+1:]=waist_mov[vds+1:]-value+np.mean(np.diff(waist_mov[:vds]))
    waist_mov=smooth(interpolate(waist_mov),40,'hanning')
    return waist_mov








def get_change_points(arr, thr_up, thr_do):
    change_points=list(np.where(arr<=thr_do)[0])+list(np.where(arr>=thr_up)[0])
    change_points.sort()
    change_points=[-1]+change_points+[len(arr)]
    change_points=np.array(change_points)+1
    return change_points


def flatten(lst):
    result = []
    for el in lst:
        if isinstance(el, list):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def remove_single_jump(arr, arr_diff, thr_up, thr_do):
    int_size=np.diff(arr)

    check=0
    check_ind=[]

    for i in range(len(int_size)):
        if int_size[i]>10:
            if check==1:
                check_ind.append(i)                   
            check=1 
        else:
            check=0
            
    for i in check_ind:
        if arr_diff[arr[i]-1]>1.4*thr_up or arr_diff[arr[i]-1]<1.4*thr_do:
            arr[i]=[arr[i]-4, arr[i]+4]
    
    arr=flatten(arr)
    
    return arr
    

def get_problem(arr):
    no_problem=[]
    
    for i in range(len(arr)-1):
        if arr[i+1]-arr[i]>10:
            no_problem=no_problem+list(range(arr[i],arr[i+1]))
    
    positions=list(range(arr[-1]))
    
    problem=sorted(set(positions)-set(no_problem))
    
    return problem
    

def corregir_vector(vector):
    vector=interpolate(vector)
    
    vds=video_separation
    value=vector[vds+1]-vector[vds]
    vector[vds+1:]=vector[vds+1:]-value
    
    for i in range(20):
        new1=smooth(vector, 20) 
        vector[0:2]=new1[0:2]
        
    diff=np.diff(vector)
    
    mean=np.mean(diff)
    std=np.std(diff)
    if std>13:
        std=std*0.9
    threshold_up=mean+2*std
    threshold_do=mean-2*std
    
    change_points_org=list(get_change_points(diff, threshold_up, threshold_do))
    
    
    
    for i in range(20):
        change_points=change_points_org.copy()
        change_points=remove_single_jump(change_points, diff, threshold_up, threshold_do)
        
        prob_int=get_problem(change_points)
        
        new1=smooth(vector, 10) 
        vector[prob_int]=new1[prob_int]
        
        diff=np.diff(vector)
        
    vector[vds+1:]=vector[vds+1:]+value
    
    diff=np.diff(vector)
    
    return vector




def corregir_error(CEfram):
    # puntos=[10, 13, 11, 14, 22, 19, 23, 20, 24, 21]
    puntos=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    for i in puntos:
        CEfram[:,i,0]=corregir_vector(CEfram[:,i,0])
        CEfram[:,i,1]=corregir_vector(CEfram[:,i,1])
        
    return CEfram



def graficar():
    puntos=[ [10, 13], [11, 14], [22, 19], [23, 20], [24, 21]]   
    for i in range(len(puntos)):
        a=puntos[i][0]
        b=puntos[i][1]
        uno=frames_moved[:,a,0].copy()
        dos=frames_moved[:,b,0].copy()
        plt.figure(i+1)
        plt.plot(uno, 'o', dos, 'o')

def graficar2():
    puntos=[ [10, 13], [11, 14], [22, 19], [23, 20], [24, 21]]   
    for i in range(len(puntos)):
        a=puntos[i][0]
        b=puntos[i][1]
        uno=keypointssmooth[:,a,0].copy()
        dos=keypointssmooth[:,b,0].copy()
        plt.figure(i+1)
        plt.plot(uno,'-',dos,'-')
            

def GetMaxMin(lista):
    puntos_extremos = []
    for sublist in lista:
        puntos_extremos.append([max(sublist), min(sublist), np.mean(sublist)])
    return puntos_extremos





def reduce_separation_vector(arr):
    new1=[]
    for i in range(20):
        new1=smooth(arr,20)
        arr[vds-19:vds+11]=new1[vds-19:vds+11]
    return arr


def reduce_separation(RSfram):
    # L=[5, 6, 7, 12, 13, 14, 19, 20, 21]
    # R=[2, 3, 4, 9, 10, 11, 22, 23, 24]
    puntos=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    
    for i in puntos:
        RSfram[:,i,0]=reduce_separation_vector(RSfram[:,i,0].copy())

    return RSfram
    

def write_csv(video_name, angles, vet):
    datos=[]
    datos.append(video_name)

    angulos_lista=flatten(angles)

    for i in angulos_lista:
        datos.append(i)
    for i in vet:
        datos.append(i)
        
    # Agregamos nuevos datos al archivo CSV
    with open('Colombia/datos.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(datos)



def complete_endpoints():
    if np.count_nonzero(frames[0])!=frames[0].size:
        ind=np.where(frames[0,:,0]==0)[0]
        to_remove = np.where(ind == 0)[0]
        ind = np.delete(ind, to_remove)
        to_remove = np.where(ind == 1)[0]
        ind = np.delete(ind, to_remove)
        to_remove = np.where(ind == 8)[0]
        ind = np.delete(ind, to_remove)
        for i in ind:
            if frames[0,opdata.joint_simmetry[i]][0]!=0:
                frames[0,i]=frames[0,opdata.joint_simmetry[i]]
            else:
                frames[5,i]=frames[5,opdata.joint_simmetry[i]]
    if np.count_nonzero(frames[-3])!=frames[-3].size:
        ind=np.where(frames[-3,:,0]==0)[0]
        to_remove = np.where(ind == 0)[0]
        ind = np.delete(ind, to_remove)
        to_remove = np.where(ind == 1)[0]
        ind = np.delete(ind, to_remove)
        to_remove = np.where(ind == 8)[0]
        ind = np.delete(ind, to_remove)
        for i in ind:
            frames[-3,i]=frames[-3,opdata.joint_simmetry[i]]
            
            
            



def fill(numframes,points):
    for i in range(numframes):
        pointx=(points[i][9][0]+points[i][12][0])/2
        pointy=(points[i][9][1]+points[i][12][1])/2
        points[i][8][0]=pointx
        points[i][8][1]=pointy

    return points



## INICIO PROGRAMA


# dia_list=['01','04','11','25']
# nvideos_list=[39, 39, 52, 48]

# l=round(random.uniform(0, 3))
# dia=dia_list[l]
# nvideo=str(round(random.uniform(1, nvideos_list[l])))

# carpeta='ArchivosJSON/Lateral/'+dia+'_03/60fps/vid'+nvideo                     #Directorio con los archivos json
#print(carpeta)
carpeta='ArchivosJSON/Lateral_nombre/Centavo/2'
#carpeta = 'Colombia/OutputVideos1_120/Lateral/lateral3_120'
files=os.listdir(carpeta)                   #Lista con todos los archivos json
frames=[]                                   #Matriz para los frames
padd_matrix=np.zeros([25,2],float)          #Matriz de relleno para los frames sin detección de personas

#Datos iniciales del video
image_height = 1080                         #Resolución Full HD para los videos laterales 
image_width = 1920
color = 0
fps=60                                      #fps del video
count=0
dist_real = 10.64                #13.7    #Distancia real que se capta en un video lateral
allpoints=25                                #Número de puntos en OpenPose
joint_pairs=list(opdata.joint_pairs.keys())

for i in files:                             #Bucle para rellenar la matriz con los datos de los archivos json
	f=open(carpeta+'/'+i)    
	data=json.load(f)
	if len(data['people'])==0:                 #Si se detecta un frame vacio se rellena con ceros
		coord=padd_matrix
	else:                                      #Si no, se leen los datos y se colocan en la matrix frames 
		pose_points=data['people'][0]['pose_keypoints_2d']
		coord=np.array(pose_points).reshape(25,3)
		coord=np.delete(coord,2,1)	           #Se elimina el puntaje de confiabilidad
	frames.append(coord)
	f.close()


frames=np.array(frames)                     #Los datos se transforman de tipo lista a array

numframes=len(frames)
posiciones=np.array(range(numframes))

complete_endpoints()





video_separation=find_separation(frames.copy())
vds=video_separation

frames[video_separation+1:]=frames[video_separation+1:]*find_ratio(frames.copy())
frames_moved=move_frames(frames.copy())

frames_corr1=corregir_error(frames_moved.copy())

frames_corr2=reduce_separation(frames_corr1.copy())

# graficar()

keypointssmooth =smoothingkeypoints(numframes,frames_corr2.copy())

# graficar2()





Angulos=calcular_angulos(keypointssmooth.copy())

AngMaxMin=GetMaxMin(Angulos)

waist_movement=get_waist_mov(frames.copy())

VET=calcular_vet2(keypointssmooth.copy(), waist_movement, dist_real)



round_kps=np.round(keypointssmooth*2)
# round_kps=np.round(frames_moved)
round_kps=np.uint(round_kps)
num_valid_frames=len(round_kps)


#gei_mat=np.zeros([image_height,image_width])
gei_mat=np.zeros([600, 600])

#GRAFICAR GEI
for i in round_kps:
    print(count)
    #img = np.full((image_height, image_width), color, dtype=np.uint8)
    img = np.full((600, 600), color, dtype=np.uint8)
    #mov_dist=i[8]-[160,140]
    contador=0
    for j in i:
        if contador!=17:
            if np.count_nonzero(j)!=0:
                cv2.circle(img,j, 5, 255, -1)
        contador+=1
        
    for j in joint_pairs:
        if np.count_nonzero(i[j[0]])!=0 and np.count_nonzero(i[j[1]])!=0:
            cv2.line(img,i[j[0]],i[j[1]],255,3) 
            
    cv2.imwrite('frames/frame'+str(count)+'.png', img)
    gei_mat=gei_mat+img
    count=count+1

gei_mat=gei_mat/(num_valid_frames)#*255) # Se guarda con float.64


# gei_mat = cv2.resize(gei_mat, (224, 224), interpolation=cv2.INTER_LINEAR)
 
name = carpeta.replace("Colombia/OutputVideos1_120/Lateral/", "")
name = name.replace("/", "_")
gei_name = "ColombiaGEI/"+name+".png"


# cv2.imwrite(gei_name, gei_mat)
# write_csv(name, AngMaxMin, VET)

# cv2.imwrite('GEI_prueba2.png', gei_mat)

# # cv2.imshow('image',gei_mat/255)         #Se muestra dividiendo para 255 rango 0-1
# cv2.imshow('image',gei_mat/255)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

    


# for i in range(allpoints):
#     plt.figure(1)
#     plt.plot(posiciones, frames_moved[:,i,0], posiciones, keypointssmooth[:,i,0])
#     plt.savefig('graphs/Parte'+str(i)+'X.png')
#     plt.clf()
#     plt.plot(posiciones, frames_moved[:,i,1], posiciones, keypointssmooth[:,i,1])
#     plt.savefig('graphs/Parte'+str(i)+'Y.png')
#     plt.clf()





