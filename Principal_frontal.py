import os


directorio = 'ArchivosJSON/Frontal_nombre'

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
        os.system("python FRONTAL.py "+carpeta)
    print('-----  END -----')
    