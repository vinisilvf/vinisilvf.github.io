import os
from tkinter import Tk, Text, mainloop, Button, Menu, filedialog as dlg, messagebox, Frame, Label, StringVar, Toplevel
import cv2
import numpy as np
import sys
import urlopen
import Processamento
import threading
import time

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
fullImage = None
baseImage = None
originalImage = None
temp_image = None
nomeArquivo = "" # Variável para armazenar o nome do arquivo
opencv_running = False #Variável para controlar o loop do OpenCV
menu_active = False #Variável para controlar a exibição do menu interativo
interactive_menu = None  # Variável global para armazenar o menu interativo

def captureImage(source):
    """Captura uma imagem de um arquivo local ou de uma URL e abre o menu interativo."""
    global fullImage, baseImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo
    global opencv_running, elementLines, tempLines, calibrationLine, menu_active

    elementLines.clear()
    tempLines.clear()
    calibrationLine.clear()
    menu_active = True  # Ativa o menu interativo somente após abrir uma imagem

    if source == 1:
        nomeArquivo = dlg.askopenfilename()
        if nomeArquivo:
            fullImage = cv2.imread(nomeArquivo)
            status_message.set("Imagem carregada de arquivo")
    else:
        status_message.set("Fonte não reconhecida")
        return

    if fullImage is not None:
        totalLines, totalColumns, rFactor = Processamento.adjustImageDimension(fullImage)
        down_points = (totalColumns, totalLines)
        # Cria a imagem base limpa (nunca modificada)
        baseImage = cv2.resize(fullImage, down_points, interpolation=cv2.INTER_LINEAR)
        # originalImage não será alterada; ela pode ser usada apenas para referência
        originalImage = baseImage.copy()
        status_message.set("Imagem carregada com sucesso")
        threading.Thread(target=run_opencv_loop, daemon=True).start()
        show_interactive_menu()  # Exibe o menu interativo junto com a imagem
    else:
        status_message.set("Erro ao carregar a imagem")

def show_interactive_menu(): #Função para exibir o menu interativo
    global interactive_menu
    interactive_menu = Toplevel()
    interactive_menu.title("Opções de Imagem")
    interactive_menu.geometry("250x150")
    interactive_menu.resizable(False, False)

    def delete_last_line(): #Função para deletar a ultima linha
        global elementLines, tempLines
        if elementLines:
            elementLines.pop()
        # Limpa também a lista de linhas temporárias
        tempLines.clear()
        update_image_cache()

    def save_image(): #Função para salvar a imagem
        if fullImage is not None:
            fileNameSave = dlg.asksaveasfilename(confirmoverwrite=False)
            if fileNameSave:
                cv2.imwrite(fileNameSave, fullImage)
        interactive_menu.destroy()

    def close_everything():
        global opencv_running, interactive_menu

        print("🛑 Encerrando o programa...")

        # 1. Parar o loop do OpenCV imediatamente
        opencv_running = False

        # 2. Fechar todas as janelas OpenCV corretamente
        if cv2.getWindowProperty("Window", cv2.WND_PROP_VISIBLE) >= 0:
            print("🔄 Fechando OpenCV...")
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Pequeno delay para garantir fechamento

        # 3. Fechar o menu interativo, se ainda estiver aberto
        if interactive_menu is not None:
            print("🔄 Fechando menu interativo...")
            try:
                interactive_menu.destroy()
            except Exception as e:
                print(f"⚠️ Erro ao fechar menu interativo: {e}")

            interactive_menu = None

    def processamento_action():  # Função para processamento de imagem
        global pixFactor, nomeArquivo, fullImage, originalImage, totalLines, totalColumns, rFactor, dFactor, elementLines
        pixFactor = 0.25041736227045075125208681135225
        #teste nome arquivo
        print("Nome Arquivo antes da chamada de processamento ==> ", nomeArquivo)
        Processamento.subImagens(fullImage, totalColumns, totalLines, rFactor, pixFactor, dFactor, nomeArquivo)
        update_image_cache()
        cv2.namedWindow("Window")  # Cria a janela
        cv2.setMouseCallback("Window", mouseActions)


    interactive_menu.protocol("WM_DELETE_WINDOW", lambda: threading.Thread(target=close_everything, daemon=True).start())

    Button(interactive_menu, text="⚙️ Processamento de Imagem", command=processamento_action).pack(pady=5, fill='x')
    Button(interactive_menu, text="🖌️ Deletar Última Linha", command=delete_last_line).pack(pady=5, fill='x')
    Button(interactive_menu, text="💾 Salvar Imagem", command=save_image).pack(pady=5, fill='x')
    Button(interactive_menu, text="❌ Fechar", command=close_everything).pack(pady=5, fill='x')


def close_interactive_menu():
    global interactive_menu, status_message

    if interactive_menu is not None:
        status_message.set("Fechando menu interativo")

        try:
            # Garantir que a janela do menu seja fechada antes de definir como None
            interactive_menu.destroy()
        except Exception as e:
            print(f"⚠️ Erro ao fechar menu interativo: {e}")

        interactive_menu = None  # Remover referência para evitar novos acessos

def update_image_cache():
    global baseImage, totalColumns, totalLines
    if baseImage is not None:
        temp_image = baseImage.copy()
        Processamento.drawLines(temp_image, calibrationLine, tempLines, elementLines, totalColumns, totalLines)
        cv2.imshow("Window", temp_image)
        cv2.waitKey(1)

def retrieve_input(textBox): # Esta função obtém o valor de entrada de um Text widget do tkinter
    global pixFactor
    inputValue = textBox.get("1.0", "end-1c")
    pixFactor = int(inputValue)
    status_message.set(f"Fator de pixel atualizado para {pixFactor}")
    textBox.quit()

def capture_distance(): # Função para capturar a distância entre dois pontos
    root = Tk()
    root.title('Informe referência em cm')
    root.geometry("300x80")
    textBox = Text(root, height=2, width=10)
    textBox.pack()
    buttonCommit = Button(root, height=1, width=10, text="Confirma", command=lambda: retrieve_input(textBox))
    buttonCommit.pack()
    mainloop()
    root.destroy()

def mouseActions(action, x, y, flags, *userdata): # Função para ações do mouse
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

def run_opencv_loop(): #Função para rodar o loop do OpenCV
    global originalImage, opencv_running
    opencv_running = True
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Window", mouseActions)

    while opencv_running:
        if originalImage is not None:
            temp_image = originalImage.copy()
            Processamento.drawLines(temp_image, calibrationLine, tempLines, elementLines, totalColumns, totalLines)
            cv2.imshow("Window", temp_image)

        key = cv2.waitKey(1)
        if key == 113 or not opencv_running:  # Sai se pressionar "q" ou o programa for encerrado
            break

        # 🚀 Nova condição para encerrar o OpenCV corretamente:
        if cv2.getWindowProperty("Window", cv2.WND_PROP_VISIBLE) < 1:
            opencv_running = False  # Para o loop
            close_interactive_menu()
            break

    cv2.destroyAllWindows()

def toggle_fullscreen(root): # Função para alternar entre tela cheia e janela
    is_fullscreen = root.attributes("-fullscreen")
    root.attributes("-fullscreen", not is_fullscreen)
    status_message.set("Tela cheia ativada" if not is_fullscreen else "Tela cheia desativada")

def toggle_video_mode(): # Função para alternar entre modo de vídeo e imagem
    global videoMode
    videoMode *= -1
    status_message.set("Modo de vídeo alternado")

def show_help():
    help_text = (
        "Comandos Disponíveis:\n\n"
        "Deletar Linha: Remove a última linha desenhada\n"
        "Salvar Imagem: Salva a imagem atual no sistema\n"
        "Alternar Modo de Vídeo: Ativa/Desativa o modo de vídeo\n"
        "Carregar Imagem: Seleciona uma imagem do arquivo\n"
        "Capturar Imagem URL: Captura imagem de uma URL ao vivo\n"
    )
    messagebox.showinfo("Ajuda - Comandos", help_text)

def exit_program(root):
    global opencv_running, interactive_menu

    print("🛑 Encerrando o programa...")

    # Parar o loop do OpenCV
    opencv_running = False

    # Fechar todas as janelas do OpenCV corretamente
    if cv2.getWindowProperty("Window", cv2.WND_PROP_VISIBLE) >= 0:
        print("🔄 Fechando OpenCV...")
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Pequeno delay para garantir que fechou

    # Fechar o menu interativo, se ainda estiver aberto
    if interactive_menu is not None and isinstance(interactive_menu, Toplevel):
        print("🔄 Fechando menu interativo...")
        try:
            interactive_menu.destroy()
        except Exception as e:
            print(f"⚠️ Erro ao fechar menu interativo: {e}")
        interactive_menu = None

    # Forçar o Tkinter a sair imediatamente
    print("🔄 Forçando atualização do Tkinter...")
    try:
        root.quit()  # Sai do mainloop() imediatamente
        root.update_idletasks()
        root.update()
    except Exception as e:
        print(f"⚠️ Erro ao atualizar root: {e}")

    # Garantir que o `mainloop()` foi realmente encerrado
    print("🛑 Finalizando Tkinter...")
    try:
        root.destroy()
    except Exception as e:
        print(f"⚠️ Erro ao destruir root: {e}")

    # FORÇAR encerramento se ainda estiver travado
    print("💀 Forçando encerramento total...")
    time.sleep(0.2)  # Pequeno delay final para garantir fechamento
    os._exit(0)



def main_menu():
    global status_message
    root = Tk()
    root.title("Menu Interativo - Processamento de Imagem")
    root.geometry("375x500")  # Define um tamanho fixo adequado para monitores modernos

    status_message = StringVar()
    status_message.set("Pronto")

    menu_bar = Menu(root)
    help_menu = Menu(menu_bar, tearoff=0)
    help_menu.add_command(label="Comandos Disponíveis", command=show_help)
    help_menu.add_command(label="Tela Cheia", command=lambda: toggle_fullscreen(root))
    menu_bar.add_cascade(label="Ajuda", menu=help_menu)
    root.config(menu=menu_bar)

    frame = Frame(root, bg="#e3f2fd")
    frame.pack(fill="both", expand=True, padx=20, pady=20)

    Button(frame, text="📂 Carregar Imagem de Arquivo", command=lambda: captureImage(1), height=2, width=40, bg="#bbdefb", fg="#0d47a1", activebackground="#90caf9", activeforeground="#0d47a1", font=("Arial", 12, "bold")).pack(pady=10)
    Button(frame, text="🌐 Capturar Imagem via URL", command=lambda: captureImage(0), height=2, width=40, bg="#c8e6c9", fg="#1b5e20", activebackground="#a5d6a7", activeforeground="#1b5e20", font=("Arial", 12, "bold")).pack(pady=10)
    Button(frame, text="🎥 Alternar Modo de Vídeo", command=toggle_video_mode, height=2, width=40, bg="#ffe0b2", fg="#e65100", activebackground="#ffcc80", activeforeground="#e65100", font=("Arial", 12, "bold")).pack(pady=10)
    Button(frame, text="❌ Sair", command=lambda: threading.Thread(target=exit_program, args=(root,), daemon= True).start(), height=2, width=40, bg="#ffccbc", fg="#b71c1c", activebackground="#ffab91", activeforeground="#b71c1c", font=("Arial", 12, "bold")).pack(pady=10)

    status_label = Label(root, textvariable=status_message, bg="#e3f2fd", fg="#0d47a1", font=("Arial", 10))
    status_label.pack(side="bottom", fill="x", pady=10)

    root.mainloop()

if __name__ == "__main__":
    main_menu()


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
