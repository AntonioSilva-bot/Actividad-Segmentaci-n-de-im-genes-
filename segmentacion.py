import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Ruta de la carpeta que contiene las imágenes
Imagenes = 'C:/Users/ansm3/Desktop/python/Segmentacion_de_imagenes/Imagenes/'

Umbralizacion_global  = 'C:/Users/ansm3/Desktop/python/Segmentacion_de_imagenes/umbralizacion_global/'
Umbralizacion_adaptativa = 'C:/Users/ansm3/Desktop/python/Segmentacion_de_imagenes/umbralizacion_adaptativa/'
Umbralizacion_otsu = 'C:/Users/ansm3/Desktop/python/Segmentacion_de_imagenes/umbralizacion_otsu/'
Segmentacion_watershed = 'C:/Users/ansm3/Desktop/python/Segmentacion_de_imagenes/Segmentacion_watershed/'
Contornos = 'C:/Users/ansm3/Desktop/python/Segmentacion_de_imagenes/Contornos/'
Gaussiano = 'C:/Users/ansm3/Desktop/python/Segmentacion_de_imagenes/Gaussiano/'
########################################################################################
Gauss_Glob = 'C:/Users/ansm3/Desktop/python/Segmentacion_de_imagenes/Gaussiano/global/'
Gauss_Ad = 'C:/Users/ansm3/Desktop/python/Segmentacion_de_imagenes/Gaussiano/adaptativa/'
Gauss_Cont = 'C:/Users/ansm3/Desktop/python/Segmentacion_de_imagenes/Gaussiano/otsu/'
Gauss_Otsu = 'C:/Users/ansm3/Desktop/python/Segmentacion_de_imagenes/Gaussiano/contornos/'

#Blanco y negro
# Blanco y negro
def umbralizacion_global(ruta_imagenes, ruta_salida):
    archivos_carpeta = os.listdir(ruta_imagenes)
    nombres_imagenes = []

    # Iteramos sobre cada archivo y agregamos solo los archivos de imagen a la lista
    for archivo in archivos_carpeta:
        # Comprobamos si el archivo tiene una extensión de imagen válida
        if archivo.endswith('.jpg') or archivo.endswith('.jpeg') or archivo.endswith('.png'):
            # Agregamos el nombre del archivo a la lista de nombres de imágenes
            nombres_imagenes.append(archivo)

    # Iteramos sobre cada imagen
    for nombre_imagen in nombres_imagenes:
        # Creamos la ruta completa del archivo de imagen
        ruta_imagen = os.path.join(ruta_imagenes, nombre_imagen)
        imagen = cv2.imread(ruta_imagen) 
        # Convertimos la imagen a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Aplicamos la umbralización global
        _, umbralizada = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)

        # Creamos la ruta completa del archivo de imagen de salida
        ruta_salida_imagen = os.path.join(ruta_salida, nombre_imagen)

        # Guardamos la imagen umbralizada
        cv2.imwrite(ruta_salida_imagen, umbralizada)

    print('¡Umbralización global finalizada!')

def segmentacion_por_contornos(ruta_imagenes, ruta_salida):
    archivos_carpeta = os.listdir(ruta_imagenes)
    nombres_imagenes = []

    # Iteramos sobre cada archivo y agregamos solo los archivos de imagen a la lista
    for archivo in archivos_carpeta:
        # Comprobamos si el archivo tiene una extensión de imagen válida
        if archivo.endswith('.jpg') or archivo.endswith('.jpeg') or archivo.endswith('.png'):
            # Agregamos el nombre del archivo a la lista de nombres de imágenes
            nombres_imagenes.append(archivo)

    # Iteramos sobre cada imagen
    for nombre_imagen in nombres_imagenes:
        # Creamos la ruta completa del archivo de imagen
        ruta_imagen = os.path.join(ruta_imagenes, nombre_imagen)
        imagen = cv2.imread(ruta_imagen) 
        # Convertimos la imagen a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Aplicamos la umbralización de la imagen
        _, umbralizada = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Buscamos los contornos en la imagen umbralizada
        contornos, _ = cv2.findContours(umbralizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibujamos los contornos sobre la imagen original
        cv2.drawContours(imagen, contornos, -1, (0, 0, 255), 2)

        # Creamos la ruta completa del archivo de imagen de salida
        ruta_salida_imagen = os.path.join(ruta_salida, nombre_imagen)

        # Guardamos la imagen con los contornos dibujados
        cv2.imwrite(ruta_salida_imagen, imagen)

    print('¡Segmentación por contornos finalizada!')



#Binarización
def umbralizacion_adaptativa(ruta_imagenes, ruta_salida):
    archivos_carpeta = os.listdir(ruta_imagenes)
    nombres_imagenes = []

    # Iteramos sobre cada archivo y agregamos solo los archivos de imagen a la lista
    for archivo in archivos_carpeta:
        # Comprobamos si el archivo tiene una extensión de imagen válida
        if archivo.endswith('.jpg') or archivo.endswith('.jpeg') or archivo.endswith('.png'):
            # Agregamos el nombre del archivo a la lista de nombres de imágenes
            nombres_imagenes.append(archivo)

    # Iteramos sobre cada imagen
    for nombre_imagen in nombres_imagenes:
        # Creamos la ruta completa del archivo de imagen
        ruta_imagen = os.path.join(ruta_imagenes, nombre_imagen)
        imagen = cv2.imread(ruta_imagen) 
        # Convertimos la imagen a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Aplicamos la umbralización adaptativa
        umbralizada = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Creamos la ruta completa del archivo de imagen de salida
        ruta_salida_imagen = os.path.join(ruta_salida, nombre_imagen)

        # Guardamos la imagen umbralizada
        cv2.imwrite(ruta_salida_imagen, umbralizada)

    print('¡Umbralización adaptativa finalizada!')

    

# Binarización de las imágenes a color
def umbralizacion_otsu(ruta_imagenes, ruta_salida):
    archivos_carpeta = os.listdir(ruta_imagenes)
    nombres_imagenes = []

    # Iteramos sobre cada archivo y agregamos solo los archivos de imagen a la lista
    for archivo in archivos_carpeta:
        # Comprobamos si el archivo tiene una extensión de imagen válida
        if archivo.endswith('.jpg') or archivo.endswith('.jpeg') or archivo.endswith('.png'):
            # Agregamos el nombre del archivo a la lista de nombres de imágenes
            nombres_imagenes.append(archivo)

    # Iteramos sobre cada imagen
    for nombre_imagen in nombres_imagenes:
        # Creamos la ruta completa del archivo de imagen
        ruta_imagen = os.path.join(ruta_imagenes, nombre_imagen)
        imagen = cv2.imread(ruta_imagen) 
        # Convertimos la imagen a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Aplicamos la umbralización Otsu
        _, umbralizada = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Creamos la ruta completa del archivo de imagen de salida
        ruta_salida_imagen = os.path.join(ruta_salida, nombre_imagen)

        # Guardamos la imagen umbralizada
        cv2.imwrite(ruta_salida_imagen, umbralizada)

    print('¡Umbralización Otsu finalizada!')


def segmentacion_watershed(ruta_carpeta):
    # Obtener una lista de todos los archivos en la carpeta
    archivos = os.listdir(ruta_carpeta)

    # Filtrar solo los archivos de imagen con extensiones conocidas
    extensiones_imagen = ['.jpg', '.jpeg', '.png', '.bmp']
    archivos_imagen = [archivo for archivo in archivos if os.path.splitext(archivo)[1].lower() in extensiones_imagen]

    # Crear una lista para almacenar todas las imágenes segmentadas
    imgs_segmentadas = []

    # Loop sobre todas las imágenes en la carpeta y aplicar la segmentación watershed a cada una
    for archivo in archivos_imagen:
        ruta_imagen = os.path.join(ruta_carpeta, archivo)
        img = cv2.imread(ruta_imagen)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, th_bin_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
        SE = np.ones((15,15), np.uint8)
        ero_img = cv2.erode(th_bin_img, SE, iterations = 1)

        ret, markers = cv2.connectedComponents(ero_img)
        markers = cv2.watershed(img, markers)
        img[markers == -1] = [255,0,0]

        # Agregar la imagen segmentada a la lista
        imgs_segmentadas.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

        print('Segmented regions: ', np.max(markers))
        plt.axis("off")
        plt.title('Segmented regions')
        plt.imshow(markers, cmap='twilight')
        plt.show()

    # Crear un collage de todas las imágenes segmentadas y mostrarlo
    num_imgs = len(imgs_segmentadas)
    if num_imgs > 0:
        collage_width = imgs_segmentadas[0].size[0]
        collage_height = int(num_imgs / 2) * imgs_segmentadas[0].size[1]
        collage = Image.new('RGB', (collage_width, collage_height))

        x_offset = 0
        y_offset = 0
        for img in imgs_segmentadas:
            if x_offset >= collage_width:
                x_offset = 0
                y_offset += imgs_segmentadas[0].size[1]
            collage.paste(img, (x_offset, y_offset))
            x_offset += imgs_segmentadas[0].size[0]

        collage.show()
    else:
        print("No se encontraron imágenes para segmentar en la carpeta.")



def ruido_gaussiano(Ruta_imagen, Gaussiano):
    # Obtener la lista de archivos de la carpeta
    archivos_carpeta_Gaussiano = os.listdir(Ruta_imagen)
    nombres_imagenes_gaussiano = []
    numero_cambios_gaussiano = 1
        # Binarización de las imágenes
    for archivo_gaussiano in archivos_carpeta_Gaussiano:
        # Comprobamos si el archivo tiene una extensión de imagen válida
        if archivo_gaussiano.endswith('.jpg') or archivo_gaussiano.endswith('.jpeg') or archivo_gaussiano.endswith('.png'):
            # Agregamos el nombre del archivo a la lista de nombres de imágenes
            nombres_imagenes_gaussiano.append(archivo_gaussiano)  

    for imagenes_gaussiano in nombres_imagenes_gaussiano:
       # Creamos la ruta completa del archivo de imagen
        ruta_imagen_gaussiano = os.path.join(Ruta_imagen, imagenes_gaussiano)
        imagen_gaussiano = cv2.imread(ruta_imagen_gaussiano) 
        ##Incertar ruido gaussiano
        imagen_gaussiano_ruido = cv2.GaussianBlur(imagen_gaussiano, (0, 0), 3)
        nombre_archivo_gaussiano = os.path.splitext(imagenes_gaussiano)[0]
        nuevo_nombre_archivo_col = f"{nombre_archivo_gaussiano}_{numero_cambios_gaussiano}_gauss.png"
        ruta_imagen_gaussiano_ruido = os.path.join(Gaussiano, nuevo_nombre_archivo_col)
        cv2.imwrite(ruta_imagen_gaussiano_ruido, imagen_gaussiano_ruido)
        numero_cambios_gaussiano += 1        
    print('¡Filtro de ruido finalizado!')



umbralizacion_global(Imagenes, Umbralizacion_global)
umbralizacion_adaptativa(Imagenes, Umbralizacion_adaptativa)
umbralizacion_otsu(Imagenes, Umbralizacion_otsu)
segmentacion_watershed(Imagenes)
segmentacion_por_contornos(Imagenes, Contornos)
ruido_gaussiano(Imagenes, Gaussiano)
ruido_gaussiano(Umbralizacion_global, Gauss_Glob)
ruido_gaussiano(Umbralizacion_adaptativa, Gauss_Ad)
ruido_gaussiano(Umbralizacion_otsu, Gauss_Cont)
ruido_gaussiano(Contornos, Gauss_Otsu)