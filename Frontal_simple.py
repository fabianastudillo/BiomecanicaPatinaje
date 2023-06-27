
import os
import cv2
import numpy as np
import json
import math
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import distance
import pandas as pd
import csv
import argparse

from mpl_toolkits.mplot3d import Axes3D


# def interpolate(function):
#     posiciones=np.array(range(numframes))
#     noerror=np.where(function > 0)
#     interp=interp1d(posiciones[noerror],function[noerror], kind='linear',bounds_error=False,fill_value="extrapolate")
#     f=interp(posiciones)
#     #final=np.where(np.isfinite(f),f,0)
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
    for i in range(allpoints):
        ceros=np.count_nonzero((poses[:,i])[:,0]==0)
        integer=ceros/frame_count
        if integer < 1:
            posex=interpolate(poses[:,i,0])
            posey=interpolate(poses[:,i,1])
            
            newposesx=smooth(posex,20,'hanning')
            newposesy=smooth(posey,20,'hanning')
           
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

def fill(numframes,points):
    for i in range(numframes):
        pointx=(points[i][9][0]+points[i][12][0])/2
        pointy=(points[i][9][1]+points[i][12][1])/2
        points[i][8][0]=pointx
        points[i][8][1]=pointy

    return points


def norma(numframes,ketpointssmooth):
    newposes=ketpointssmooth
    for i in range(numframes):
        x1=ketpointssmooth[i][0][0]
        y1=ketpointssmooth[i][0][1]

        x2=(ketpointssmooth[i][11][0]+ketpointssmooth[i][14][0])/2
        y2=(ketpointssmooth[i][11][1]+ketpointssmooth[i][14][1])/2
        
        distancia=(distance.euclidean((x1,y1),(x2,y2)))/300
        
        for n in range(8):  
            newx=(ketpointssmooth[i][n][0]-ketpointssmooth[i][8][0])/distancia
            newy=(ketpointssmooth[i][n][1]-ketpointssmooth[i][8][1])/distancia
            
            newposes[i][n][0]=newx
            newposes[i][n][1]=newy

        for n in range(9,25):  
            newx=(ketpointssmooth[i][n][0]-ketpointssmooth[i][8][0])/distancia
            newy=(ketpointssmooth[i][n][1]-ketpointssmooth[i][8][1])/distancia
           
            newposes[i][n][0]=newx
            newposes[i][n][1]=newy

    finalposes=fill(numframes, newposes)
    
    return finalposes
    
def position(numframes,matrix):
    newmatrix=[]
    for i in matrix:
        newx=i[8]/2-[300,200]#[600,200]
        positionnew=[]
        for j in i:
            new=j-newx
            positionnew.append(new)
        newmatrix.append(np.array(positionnew))
    return np.array(newmatrix)


def getAngle(a, b, c): # angulo de C a A en sentido horario
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang



def calcular_angulos(CAfram):
    Rfoot_ex=[]
    Lfoot_ex=[]
    Rfoot_in=[]
    Lfoot_in=[]
    Rpush_in=[]
    Lpush_in=[]
    Rpush_ex=[]
    Lpush_ex=[]
    Rankle=[]
    Lankle=[]
    Legs=[]
    
    for i in range(len(CAfram)):
        angle1=getAngle(CAfram[i,22], CAfram[i,11], CAfram[i,11]-[100,0])
        angle2=getAngle(CAfram[i,14]+[100,0], CAfram[i,14], CAfram[i,19])
        angle3=180-angle1
        angle4=180-angle2
        angle5=getAngle(CAfram[i,22], CAfram[i,2], CAfram[i,2]+[0,100])
        angle6=getAngle(CAfram[i,5]+[0,100], CAfram[i,5], CAfram[i,19])
        angle7=0
        angle8=0
        angle9=getAngle(CAfram[i,11]-[100,0], CAfram[i,11], CAfram[i,10])  
        angle10=getAngle(CAfram[i,13], CAfram[i,14], CAfram[i,14]+[100,0])
        
        #si el angulo 7 u 8 son 0 eso significa que el tobillo esta por encima de la rodilla
        
        if angle1>180:
            angle1=0
            angle3=0
        
        if angle2>180:
            angle2=0
            angle4=0
        
        if angle5>=180:
            angle7=360-angle5
            angle5=0
            angle9=0
        if angle6>=180:
            angle8=360-angle6
            angle6=0
            angle10=0
        
        if angle9>=180:
            angle9=0
        if angle10>=180:
            angle10=0
            
        angle11=getAngle(CAfram[i,10], CAfram[i,8], CAfram[i,13])       #Ángulo entre las piernas F
        
        if angle11 >= 180:           #Este angulo es absoluto 
            angle11=360-angle11
        
        Rfoot_ex.append(angle1)
        Lfoot_ex.append(angle2)
        Rfoot_in.append(angle3)
        Lfoot_in.append(angle4)
        Rpush_in.append(angle5)
        Lpush_in.append(angle6)
        Rpush_ex.append(angle7)
        Lpush_ex.append(angle8)
        Rankle.append(angle9)
        Lankle.append(angle10)
        Legs.append(angle11)
        
    return [Rfoot_ex, Lfoot_ex, Rfoot_in, Lfoot_in, Rpush_in, Lpush_in, Rpush_ex, Lpush_ex, Rankle, Lankle, Legs]
        

def vector_euclidean_distance(vec1, vec2):
    # Comprobamos que los vectores tengan la misma longitud
    if len(vec1) != len(vec2):
      raise ValueError("Los vectores deben tener la misma longitud")
    # Inicializamos el vector de distancias a una lista vacía
    distances = []
    # Iteramos sobre los elementos de los vectores y añadimos
    # a la lista distances la distancia euclidiana entre ellos
    for i in range(len(vec1)):
      distancia = distance.euclidean(vec1[i], vec2[i])
      distances.append(distancia)
    # Devolvemos el vector de distancias
    return np.array(distances)

def dist_ciclo(DCfram):
    dist_horizontal= abs(DCfram[:,11,0]-DCfram[:,14,0])#distancia horizontal
    dist_euclidiana= vector_euclidean_distance(DCfram[:,11], DCfram[:,14])
    return [dist_horizontal, dist_euclidiana]



def get_height(GHfram):
    
    arr1=[]
    arr2=[]
    arr3=[]
    
    for i in range(len(GHfram)):
        if frames[i,11,1]!=0 and frames[i,14,1]!=0 and frames[i,1,1]!=0:
            arr1.append(GHfram[i,11,1])
            arr2.append(GHfram[i,14,1])
            arr3.append(GHfram[i,1,1])
    min_array = []
    
    for a, b in zip(arr1, arr2):
        min_array.append(max(a, b))
    
    min_array=np.array(min_array)
    arr3=np.array(arr3)
    arr4=abs(arr3-min_array)
    
    altura=sum(arr4)/len(arr4)
    
    return altura



def calcular_vet(CVfram, dr, pix):
    amp_front_max=[]
    Rfoot_floor_sep=[]
    Lfoot_floor_sep=[]
    Rmax_floor_sep=[]
    Lmax_floor_sep=[]
    
    # feet_separation=dist_ciclo(CVfram)[0]
    shoulder_distance=vector_euclidean_distance(CVfram[:,2], CVfram[:,5])
    shoulder_distance_mean=np.mean(shoulder_distance)*(dr/pix)
    
    arrR = CVfram[:,11,0]
    arrL = CVfram[:,14,0]
    
    diff_h = arrL-arrR
    peaks,_ = find_peaks(diff_h, height=30, distance=20)
    valleys,_ = find_peaks(-diff_h, distance=20)
    
    meanR =np.mean(arrR)
    meanL =np.mean(arrL)
    aux = arrL-meanL+arrR-meanR
    
    amp_front_max = diff_h[peaks]*np.sign(aux[peaks])
    
    amp_max_FR = abs(amp_front_max[amp_front_max < 0])
    amp_max_FL = abs(amp_front_max[amp_front_max > 0])
    
    if len(amp_max_FL)==0:
        amp_max_FL = amp_max_FR
    if len(amp_max_FR)==0:
        amp_max_FR = amp_max_FL
        
    amp_max_FR = sum(amp_max_FR)/len(amp_max_FR)*(dr/pix)
    amp_max_FL = sum(amp_max_FL)/len(amp_max_FL)*(dr/pix)
        
    
    amp_front_min = diff_h[valleys]*np.sign(aux[valleys])

    amp_min_FR = abs(amp_front_min[amp_front_min > 0])
    amp_min_FL = abs(amp_front_min[amp_front_min < 0])
    
    if len(amp_min_FL)==0:
        amp_min_FL = amp_min_FR
    if len(amp_min_FR)==0:
        amp_min_FR = amp_min_FL
    
    amp_min_FR = sum(amp_min_FR)/len(amp_min_FR)*(dr/pix)
    amp_min_FL = sum(amp_min_FL)/len(amp_min_FL)*(dr/pix)    
    
    
    dist_vertical= CVfram[:,11,1]-CVfram[:,14,1]
    pie1=np.zeros(len(dist_vertical))
    pie2=np.zeros(len(dist_vertical))
    
    ind1=np.where(dist_vertical>=0)[0]
    ind2=np.where(dist_vertical<0)[0]
    
    pie1[ind1]=dist_vertical[ind1]
    pie2[ind2]=-dist_vertical[ind2]
    
    if dist_vertical[0]<0:
        Lfoot_floor_sep=pie2
        Rfoot_floor_sep=pie1
    else:
        Lfoot_floor_sep=pie1
        Rfoot_floor_sep=pie2
        
    Rmax_floor_sep=max(Rfoot_floor_sep)*(dr/pix)
    Lmax_floor_sep=max(Lfoot_floor_sep)*(dr/pix)
    
    return [amp_max_FR, amp_max_FL,  amp_min_FR, amp_min_FL, shoulder_distance_mean, Rmax_floor_sep, Lmax_floor_sep]


    
def GetMaxMin(lista):
    puntos_extremos = []
    for sublist in lista:
        sublist_no_cero = list(filter(lambda x: x != 0, sublist))
        # puntos_extremos.append([max(sublist), min(sublist_no_cero)])
        puntos_extremos.append([max(sublist_no_cero ), min(sublist_no_cero ), np.mean(sublist_no_cero )])
    return puntos_extremos


def flatten(lst):
    result = []
    for el in lst:
        if isinstance(el, list):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result



def write_csv(video_name, angles, vet):
    datos=[]
    datos.append(video_name)

    angulos_lista=flatten(angles)

    for i in angulos_lista:
        datos.append(i)
    for i in vet:
        datos.append(i)
        
    # Agregamos nuevos datos al archivo CSV
    with open('SVM/datos2.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(datos)


def get_dist_real(Vid_name):
    data = pd.read_csv("SVM/datos.csv").set_index("Video")
    # Extrae una celda específica utilizando el índice
    nombre_fila = Vid_name
    nombre_columna = "height"
    valor_celda = data.loc[nombre_fila, nombre_columna]
    
    return valor_celda













def get_change_points(arr, thr_up, thr_do):
    change_points=list(np.where(arr<=thr_do)[0])+list(np.where(arr>=thr_up)[0])
    change_points.sort()
    change_points=[-1]+change_points+[len(arr)]
    change_points=np.array(change_points)+1
    return change_points


def remove_single_jump(arr, arr_diff, thr_up, thr_do):
    int_size=np.diff(arr)

    check=0
    check_ind=[]

    for i in range(len(int_size)):
        if int_size[i]>40:
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
        if arr[i+1]-arr[i]>40:
            no_problem=no_problem+list(range(arr[i],arr[i+1]))
    
    positions=list(range(arr[-1]))
    
    problem=sorted(set(positions)-set(no_problem))
    
    return problem
    

# def corregir_vector(vector):
#     vector=interpolate(vector)
    
#     for i in range(20):
#         new1=smooth(vector, 20) 
#         vector[0:2]=new1[0:2]
        
#     diff=np.diff(vector)
    
#     mean=np.mean(diff)
#     std=np.std(diff)
#     if std>13:
#         std=std*0.9
#     threshold_up=mean+2*std
#     threshold_do=mean-2*std
    
#     change_points_org=list(get_change_points(diff, threshold_up, threshold_do))
    
    
    
#     for i in range(20):
#         change_points=change_points_org.copy()
#         change_points=remove_single_jump(change_points, diff, threshold_up, threshold_do)
        
#         prob_int=get_problem(change_points)
        
#         new1=smooth(vector, 10) 
#         vector[prob_int]=new1[prob_int]
        
#         diff=np.diff(vector)
            
#     diff=np.diff(vector)
    
#     return vector



def corregir_vector(vector):
    vector=interpolate(vector)
    
    for i in range(20):
        new1=smooth(vector, 20) 
        vector[0:2]=new1[0:2]
    
    for i in range(60):
        new1=smooth(vector, 60)
        indice=abs(vector-new1).argmax()    
        vector[indice]=new1[indice]
    
    return vector


def corregir_error(CEfram):
    #puntos=[8]
    puntos=[0, 1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    for i in puntos:
        CEfram[:,i,0]=corregir_vector(CEfram[:,i,0])
        CEfram[:,i,1]=corregir_vector(CEfram[:,i,1])
        
    return CEfram



def graficar_tray(GTfram, nombre):
    plt.figure(figsize=(5, 6))
    pos=posiciones[::-1]

    plt.plot(GTfram[:,11,0],pos,'o')
    plt.plot(GTfram[:,14,0],pos,'o')
    plt.xlim(0, 600)

    plt.axis('off')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.savefig('Trayectoria/'+nombre+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()


# parser = argparse.ArgumentParser()
# # Añadimos un argumento llamado "mensaje"
# parser.add_argument("mensaje", help="Un mensaje para imprimir")
# # Parseamos los argumentos
# args = parser.parse_args()
# carpeta=args.mensaje


# carpeta='Colombia/OutputVideos1_120/Frontal/frontal3V2zoom2'
carpeta = 'ArchivosJSON/Frontal_nombre/Ambar/0'
files=os.listdir(carpeta)
frames=[]
padd_matrix=np.zeros([25,2],float)
for i in files:
	f=open(carpeta+'/'+i)
    
	data=json.load(f)
	if len(data['people'])==0:
		coord=padd_matrix
	else:
		pose_points=data['people'][0]['pose_keypoints_2d']
		coord=np.array(pose_points).reshape(25,3)
		coord=np.delete(coord,2,1)	
	frames.append(coord)
	f.close()


# name = carpeta.replace("Colombia/OutputVideos1_120/Frontal/",'')
name = carpeta.replace('ArchivosJSON/Frontal_nombre/','')
name = name.replace("/", "_")


frames=np.array(frames)
numframes=len(frames)
allpoints=25


# frames1 = corregir_error(frames.copy())
ketpointssmooth=smoothingkeypoints(numframes,frames.copy())


normalizar=norma(numframes,ketpointssmooth.copy()) 
newposition=position(numframes,normalizar.copy()) 

# dist_real = get_dist_real('lateral3_120')
dist_real = get_dist_real(name)
dist_pixels = get_height(newposition.copy())





##########################
Angulos = calcular_angulos(newposition.copy())
AngMaxMin = GetMaxMin(Angulos)

VET=calcular_vet(newposition.copy(), dist_real, dist_pixels)
##########################


round_kps=(np.round(abs(newposition[0:numframes])))
# round_kps=(np.round(abs(frames1[0:numframes])))
round_kps=np.uint(round_kps)


JOINT_PAIRS_MAP_ALL = {(0, 15): {'joint_names': ('Nose', 'REye')},
                       (0, 16): {'joint_names': ('Nose', 'LEye')},
                       (1, 0): {'joint_names': ('Neck', 'Nose')},
                       (1, 2): {'joint_names': ('Neck', 'RShoulder')},
                       (1, 5): {'joint_names': ('Neck', 'LShoulder')},
                       (1, 8): {'joint_names': ('Neck', 'MidHip')},
                       (2, 3): {'joint_names': ('RShoulder', 'RElbow')},
                       (3, 4): {'joint_names': ('RElbow', 'RWrist')},
                       (5, 6): {'joint_names': ('LShoulder', 'LElbow')},
                       (6, 7): {'joint_names': ('LElbow', 'LWrist')},
                       (8, 9): {'joint_names': ('MidHip', 'RHip')},
                       (8, 12): {'joint_names': ('MidHip', 'LHip')},
                       (9, 10): {'joint_names': ('RHip', 'RKnee')},
                       (10, 11): {'joint_names': ('RKnee', 'RAnkle')},
                       (11, 22): {'joint_names': ('RAnkle', 'RBigToe')},
                       (11, 24): {'joint_names': ('RAnkle', 'RHeel')},
                       (12, 13): {'joint_names': ('LHip', 'LKnee')},
                       (13, 14): {'joint_names': ('LKnee', 'LAnkle')},
                       (14, 19): {'joint_names': ('LAnkle', 'LBigToe')},
                       (14, 21): {'joint_names': ('LAnkle', 'LHeel')},
                       (15, 17): {'joint_names': ('REye', 'REar')},
                       (16, 18): {'joint_names': ('LEye', 'LEar')},
                       (19, 20): {'joint_names': ('LBigToe', 'LSmallToe')},
                       (22, 23): {'joint_names': ('RBigToe', 'RSmallToe')}}

b=list(JOINT_PAIRS_MAP_ALL.keys())

image_height = 1080
image_width = 1920
color = 0

count=0

gei_mat=np.zeros([1080, 1920])
# gei_mat=np.zeros([600, 600])

for i in round_kps:
    img = np.full((1080, 1920), color, dtype=np.uint8)
    # img = np.full((600, 600), color, dtype=np.uint8)
    mov_dist=[0,-35]
    for j in i:
        if np.count_nonzero(j)!=0:
            cv2.circle(img,j-mov_dist, 5, 255, -1)
    for j in b:
        if np.count_nonzero(i[j[0]])!=0 and np.count_nonzero(i[j[1]])!=0:
            cv2.line(img,i[j[0]]-mov_dist,i[j[1]]-mov_dist,255,3) 
    #cv2.imwrite('lateralfr/frame'+str(count)+'.png', img)
    cv2.imwrite('frames/frame'+str(count)+'.png', img)
    gei_mat=gei_mat+img
    count=count+1

gei_mat=gei_mat/(numframes)#*255) #Se guarda con float.64

# gei_mat = cv2.resize(gei_mat, (224, 224), interpolation=cv2.INTER_LINEAR)
 

gei_name = "ColombiaGEI/"+name+".png"

# cv2.imwrite(gei_name, gei_mat)
# write_csv(name, AngMaxMin, VET)


 

# cv2.imshow('image',matriz_redimensionada/255)
# cv2.imshow('image',gei_mat/255)  #Se muestra dividiendo para 255 rango 0-1
# cv2.waitKey(0)
# cv2.destroyAllWindows()




posiciones=np.array(range(numframes))  


# graficar_tray(newposition, name)

# for i in range(allpoints):
#     #plt.figure(2)
#     plt.plot(round_kps[:,i,0])
#     #print(round_kps[:,i,0])
#     #plt.ylim(400, 800)
#     plt.savefig('lateralgr/Parte'+str(i)+'X.png')
#     plt.clf()
#     plt.plot(round_kps[:,i,1])
#     #plt.ylim(50, 300)
#     plt.savefig('lateralgr/Parte'+str(i)+'Y.png')
#     plt.clf()
#     plt.close



