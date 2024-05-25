from tkinter import Tk, Text, mainloop, Button
from tkinter import filedialog as dlg
import cv2
import numpy as np
import sys
import Processamento

# Lists to store the Lines coordinators
elementLines = []
calibrationLine = []
tempLines = []
clicked = 0
totalLines = 0
totalColumns = 0
pixFactor = 0.247525
dFactor = 1
auxFactor = 1
videoMode = -1


# Esta função obtém o valor de entrada de um Text widget do tkinter
def retrieve_input(textBox):
    global pixFactor
    inputValue = textBox.get("1.0", "end-1c")
    textBox.quit()
    pixFactor = int(inputValue)


# Cria uma janela Tkinter para o usuário inserir um valor de referência em centímetros. O valor é capturado e usado para atualizar o pixFactor.
def capture_distance():
    root = Tk()
    root.title('Informe referência em cm')
    root.geometry("300x80")
    textBox = Text(root, height=2, width=10)
    textBox.pack()
    buttonCommit = Button(root, height=1, width=10, text="confirma", command=lambda: retrieve_input(textBox))
    # command=lambda: retrieve_input() >>> just means do this when i press the button
    buttonCommit.pack()
    mainloop()
    root.destroy()


# Esta função gerencia as ações do mouse na janela do OpenCV. Dependendo do tipo de ação do mouse (botão pressionado, liberado, movimento),
# ela atualiza as listas de coordenadas para linhas e a linha de calibração.
# function which will be called on mouse input
def mouseActions(action, x, y, flags, *userdata):
    # Referencing global variables
    global elementLines, tempLines, originalImage, clicked, totalLines, totalColumns, calibrationLine, pixFactor, dFactor, auxFactor
    # Mark the top left corner when left mouse button is pressed

    if action == cv2.EVENT_LBUTTONDOWN:
        if clicked == 3:
            calibrationLine.append((x, y, clicked))
            calibrationLine.append((x, y, clicked))
        else:
            clicked = 1
            tempLines = [(y, clicked)]
            # When left mouse button is released, mark bottom right corner
    elif action == cv2.EVENT_LBUTTONUP:
        if clicked == 3:
            calibrationLine[1] = (x, y, clicked)
            xi, yi, c = calibrationLine[0]
            xf, yf, c = calibrationLine[1]
            calibrationLine = []
            clicked = 0
            try:
                dX = xf * (1 / auxFactor) - xi * (1 / auxFactor)
                dY = yf * (1 / auxFactor) - yi * (1 / auxFactor)
                dFactor = (dX ** 2 + dY ** 2) ** 0.5
                print(f'Medida em Pixel => {dFactor}')
                capture_distance()
                pixFactor = pixFactor / dFactor
            except ValueError:
                pixFactor = 1
        else:
            elementLines.append((y, clicked))
        clicked = 0
        # Draw the rectangle
    elif action == cv2.EVENT_RBUTTONDOWN:
        clicked = 2
        tempLines = [(x, clicked)]
    elif action == cv2.EVENT_RBUTTONUP:
        elementLines.append((x, clicked))
        clicked = 0
    elif action == cv2.EVENT_MOUSEMOVE:
        if clicked != 0:
            tempLines = []
            if clicked == 3 and len(calibrationLine) > 1:
                calibrationLine[1] = (x, y, clicked)
            if clicked == 1:
                tempLines = [(y, clicked)]
            if clicked == 2:
                tempLines = [(x, clicked)]


# Esta função captura uma imagem de diferentes fontes: arquivo local, URL ou um arquivo específico.
# A imagem é então ajustada para caber na tela e redimensionada. Retorna a imagem completa, a imagem redimensionada,
# linhas e colunas totais, fator de redimensionamento e nome do arquivo.
def captureImage(source):
    fileName = ''
    if (source == 1):
        fileName = dlg.askopenfilename()  # .asksaveasfilename(confirmoverwrite=False)
        if fileName != '':
            print(fileName)
            fullImage = cv2.imread(fileName)
    elif (source == 0):
        with urlopen('http://10.14.38.133:8080/shot.jpg') as url:
            imgResp = url.read()
        imgNp = np.array(bytearray(imgResp), dtype=np.uint8)  # Numpy to convert into a array
        fullImage = cv2.imdecode(imgNp, -1)  # Finally decode the array to OpenCV usable format ;)
    elif (source == 2):
        fullImage = cv2.imread('D:/TCC/ovo_base.jpg')

    # fullImage = cv2.rotate(fullImage, cv2.ROTATE_90_CLOCKWISE)
    totalLines, totalColumns, rFactor = Processamento.adjustImageDimension(fullImage)  # Adjust full image to fit on screem
    down_points = (totalColumns, totalLines)
    originalImage = cv2.resize(fullImage, down_points, interpolation=cv2.INTER_LINEAR)

    return (fullImage, originalImage, totalLines, totalColumns, rFactor, fileName)


# Programa Principal
# Ele inicializa a janela do OpenCV, define o callback do mouse, captura uma imagem inicial e entra em um loop onde:
# Exibe a imagem e desenha as linhas nela. E Verifica as teclas pressionadas para executar diferentes ações.
if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib.request import urlopen

cv2.namedWindow("Window")  # Create a named window
cv2.setMouseCallback("Window", mouseActions)  # highgui function called when mouse events occur
fullImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo = captureImage(2)  # Read Images
auxFactor = rFactor

while True:
    # Display the image
    if videoMode == 1:
        fullImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo = captureImage(0)

    image = originalImage.copy()
    image = Processamento.drawLines(image, calibrationLine, tempLines, elementLines, totalColumns, totalLines)

    cv2.imshow("Window", image)
    k = cv2.waitKey(1)

    if (k == 113):  # 'q' sair do sistema
        break

    if (k == 100):  # 'd' deletar a última linha
        if len(elementLines) > 0:
            del elementLines[-1]
        tempLines = []
        clicked = 0

    if (k == 112):  # 'p' ativar o processamento de imagem
        pixFactor = 0.25041736227045075125208681135225
        print("Nome Arquivo antes da chamada de processamento ==> ", nomeArquivo)
        Processamento.subImagens(elementLines, fullImage, totalColumns, totalLines, rFactor, pixFactor, dFactor, nomeArquivo)
        cv2.namedWindow("Window")  # Create a named window
        cv2.setMouseCallback("Window", mouseActions)
        fullImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo = captureImage(1)

    if (k == 99):  # 'c' capturar uma nova imagem
        videoMode = -1
        fullImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo = captureImage(0)

    if (k == 102):  # 'f' ler imagem de arquivo
        fullImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo = captureImage(1)

    if (k == 120):  # 'x' iniciar calibração pixFactor
        clicked = 3
        calibrationLine = []
    if (k == 118):  # 'v' alternar modo de vídeo
        videoMode = videoMode * -1

    if (k == 115):  # 's' salvar a imagem em um arquivo
        fileNameSave = dlg.asksaveasfilename(confirmoverwrite=False)
        if fileNameSave != '':
            cv2.imwrite(fileNameSave, fullImage)

cv2.destroyAllWindows()

'''
def subImagens(imagem):
    imagens = {
        1: [355, 65, 555, 220],
        2: [365, 220, 555, 380],
        3: [365, 380, 555, 545],
        4: [365, 545, 555, 715],
        5: [365, 715, 555, 895],
        6: [365, 875, 555, 1055],
        7: [555, 220, 770, 380],
        8: [555, 220, 770, 380],
        9: [555, 380, 770, 545],
        10: [555, 545, 770, 715],
        11: [555, 715, 770, 895],
        12: [555, 875, 770, 1055],
        13: [770, 380, 965, 545],
        14: [770, 220, 965, 380],
        15: [770, 380, 965, 545],
        16: [770, 545, 965, 715],
        17: [770, 715, 965, 895],
        18: [770, 875, 965, 1055],
        19: [965, 545, 1150, 715],
        20: [965, 220, 1150, 380],
        21: [965, 380, 1150, 545],
        22: [965, 545, 1150, 715],
        23: [965, 715, 1150, 895],
        24: [965, 875, 1150, 1055]};
    i=1;
    while True:
        recorte = imagens[i];
        (xi, yi, xf, yf) = recorte;
        print(f'{xi}, {xf}, {yi}, {yf}')
        subImagem = imagem[xi:xf, yi:yf]
        resultado = Processamento.process(subImagem, 4)
        print(resultado)
        mensagem = f'Recorte {i}'
        cv2.imshow(mensagem, subImagem)
        k = cv2.waitKey(0)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            i+=1;

def click_event(event, x, y, flags, param):
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
        cv2.imshow('image', img)
    if event == cv2.EVENT_RBUTTONDBLCLK:
        img = cv2.imread(img_path)
        print("cleaned")
        cv2.imshow('image', img)

def drawImage():
    while True:
        cv2.setMouseCallback('Image', click_event)
        cv2.imshow('IPWebcam', img)
        k=cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            return k



if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib.request import urlopen
img_path = 'ImagemTeste.jpg'
global img
img = cv2.imread(img_path)
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.namedWindow("image")
cv2.setMouseCallback('image', click_event)



while True:
    ##with urlopen('http://10.14.30.138:8080/shot.jpg') as url:
    ##    imgResp = url.read()

        # Numpy to convert into a array
    ##imgNp = np.array(bytearray(imgResp), dtype=np.uint8)

    # Finally decode the array to OpenCV usable format ;)
    ##img = cv2.imdecode(imgNp, -1)

    # put the image on screen

    cv2.imshow('image', img)
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        cv2.imwrite("ImagemTeste.jpg", img)
        subImagens(img)

    # To give the processor some less stress
    # time.sleep(0.1)
cv2.destroyAllWindows()


cam = cv2.VideoCapture('http://10.14.30.138:4747/mjpegfeed')

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img = frame
        #img = cv2.rotate(img, cv2.ROTATE_90_)
        height, width = img.shape[:2]
        colors = [(255, 0, 0)]

        cv2.imwrite("ImagemTeste.jpg", img)


        cv2.imshow("Quadro", img)
        cv2.waitKey(1)


cam.release()

cv2.destroyAllWindows()
'''
'''
Trecho de código de tratamento da rede yolo
        Class_names = []
        with open("coco.names", "r") as f:
            Class_names = [cname.strip() for cname in f.readlines()]

        net1 = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        net2 = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

        model1 = cv2.dnn_DetectionModel(net1)
        model1.setInputParams(size=(416, 416), scale=1 / 255)

        model2 = cv2.dnn_DetectionModel(net2)
        model2.setInputParams(size=(416, 416), scale=1 / 255)

        print(height, " - ", width);
        linhas = [0, 270, 565, height]
        colunas = [0, 300, 544, width]

        # for i in range(3):
        #     for j in range(3):
        #         newImg = img[linhas[i]:linhas[i+1],colunas[j]:colunas[j+1]]
        newImg = img
        classes, scores, boxes = model1.detect(newImg, 0.000001, 0.000002)

        box = zip(boxes);
        print("Caixas ==> ", boxes)

        for (classid, score, box) in zip(classes, scores, boxes):
            # box[0]+=colunas[j]
            # box[1]+=linhas[i]
            (largura, altura) = box[2:4]
            # if (altura < 600)  and (classid[0]==32):
            print("Altura x Largura  = (%d x %d) Razao de Aspecto = %.4f" % (altura, largura, altura / largura));
            color = colors[int(classid) % len(colors)]
            # label = f"{Class_names[classid[0]]} : {score}"
            label = f"{score}"
            cv2.rectangle(img, box, color, 2)
            cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # for i in range(3):
        #     for j in range(3):
        #         newImg = img[linhas[i]:linhas[i+1],colunas[j]:colunas[j+1]]

        newImg = img

        classes, scores, boxes = model2.detect(newImg, 0.000001, 0.000002)

        for (classid, score, box) in zip(classes, scores, boxes):
            # box[0]+=colunas[j]
            # box[1]+=linhas[i]
            (largura, altura) = box[2:4]
            # if (altura < 600)  and (classid[0]==32):
            print("Altura x Largura  = (%d x %d) Razao de Aspecto = %.4f" % (altura, largura, altura / largura));
            color = colors[int(classid) % len(colors)]
            # label = f"{Class_names[classid[0]]} : {score}"
            label = f"{score}"
            cv2.rectangle(img, box, color, 2)
            cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

'''

# #path = dlg.askopenfilename(title = "Selecione o ovo a ser avaliado",
# #                           filetypes=(('jpg files', '*.jpg'), ('jpeg files', '*.jpeg')))
#
# #img=cv2.imread(path)
# #colors = [(0,255,255),(255,255,0),(0,255,0),(255,0,0)]
# colors = [(255,0,0)]
#
# folder_selected = dlg.askdirectory() + "/";
# files = os.scandir(folder_selected)
#
#
# Class_names = []
# with open("coco.names", "r") as f:
#     Class_names = [cname.strip() for cname in f.readlines()]
#
# #cv2.imshow("Original", imagem)
#
# #cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
# #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#
# #net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
#
# model = cv2.dnn_DetectionModel(net)
# model.setInputParams(size=(416,416), scale = 1/255)
#
# for T in files:
#     if T.name.endswith(('.jpg','.jpeg')):
#         imgFile = folder_selected + T.name
#         img = cv2.imread(imgFile)
#         #contador = 1;
#
#         classes, scores, boxes = model.detect(img, 0.000001,0.000002)
#
#         for (classid, score, box) in zip(classes,scores,boxes):
#             (largura,altura) = box[2:4]
#             #if (altura < 600)  and (classid[0]==32):
#             print("Altura x Largura  = (%d x %d) Razao de Aspecto = %.4f" % (altura, largura, altura / largura));
#             color = colors[int(classid)%len(colors)]
#             label = f"{Class_names[classid[0]]} : {score}"
#             cv2.rectangle(img,box,color,2)
#             cv2.putText(img,label,(box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
#
#         cv2.imshow("Quadro", img)
#         cv2.waitKey(0)
#
#                 #imgFile = folder_selected + 'Processed_' + f.name
#
#                 #cv2.imwrite(imgFile, img)
#
#
#
# #
# cv2.destroyAllWindows()
