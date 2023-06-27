import os


directorio = 'ArchivosJSON/Lateral_nombre'

#directorio = 'Colombia/OutputVideos1_120/Lateral'

subdirectorios=os.listdir(directorio) 

for i in range(len(subdirectorios)):
    subdirectorios[i]=directorio+'/'+subdirectorios[i]

for i in subdirectorios:
    subsubdirectorios=os.listdir(i)
    print(i)
    print('----------')
    for j in subsubdirectorios:
        carpeta=i+'/'+j
        print(carpeta)
        os.system("python get_video_data.py "+carpeta)
    print('-----  END -----')
    
# os.system("python get_video_data.py "+directorio)