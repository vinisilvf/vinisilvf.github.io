from math import atan, degrees
from tkinter import filedialog as dlg
import cv2
import numpy as np
import pandas as pd
import os
import math
import multiprocessing
from openpyxl import Workbook  # pip install openpyxl
from openpyxl import load_workbook
from skimage.measure import label, regionprops_table
from skimage.transform import resize
from skimage.measure import find_contours
from skimage.color import rgb2hsv
from skimage.morphology import disk
from skimage.filters import median
from scipy.optimize import curve_fit  # Lib p/ fazer ajuste polinomial
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def check_and_create_directory_if_not_exist(path_directory):
    if not os.path.exists(path_directory):
        os.makedirs(path_directory)
        print(f"The path {path_directory} is created!")

# Esta função ajusta as dimensões de uma imagem para garantir que ela
# não exceda 1800 pixels em largura ou altura, mantendo a proporção.
# im > imagem de entrada
def adjustImageDimension(im):
    tLines, tColumns, c = im.shape
    if tLines > tColumns:
        rFactor = 1800 / tLines
        tLines = 1800
        tColumns = int(tColumns * rFactor)
    else:
        rFactor = 1800 / tColumns
        tColumns = 1800
        tLines = int(tLines * rFactor)
    return (tLines, tColumns, rFactor)

# extraída da parte interna do (for egg in posEggs) antigo codigo da função subimagens
def process_single_egg(egg, img_bytes, shape, factor, pixFactor, dFactor, processed_path, plot_results_path, nFile, path_file_name):
    from Processamento import process, create_sheet, disc_slices_curve_fit, polynomial_curv_3, polynomial_curv_5, polynomial_curv_7, polynomial_curv_9, polynomial_curv_11  # reimportação dentro do subprocesso (necessária p/ Windows)
    imgProcess = np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape).copy()
    (gr, ll, col, lin, larg, alt) = egg
    lIni = int(lin - round(alt * 0.075))
    lFin = int(lin + round(alt * 1.075))
    cIni = int(col - round(larg * 0.075))
    cFin = int(col + round(larg * 1.075))
    recImage = imgProcess[lIni:lFin, cIni:cFin].copy()
    # Criar pasta do ovo para gráficos
    egg_num = str(nFile)
    egg_folder_fit_plot_path = Path(plot_results_path, egg_num)
    check_and_create_directory_if_not_exist(egg_folder_fit_plot_path)

    resultado = process(recImage, 1, pixFactor, egg_num, path_file_name, egg_folder_fit_plot_path)

    # print(f"\n\nRESULTADO Processamento da imagem :\n {resultado}\n\n")
    imgProc, a, b, c, d, v, area, pAi, pAf, pBi, pBf = resultado
    cSEx = int(min(pAi[0], pAf[0], pBi[0], pBf[0]))
    cSEy = int(min(pAi[1], pAf[1], pBi[1], pBf[1]))

    cIDx = int(max(pAi[0], pAf[0], pBi[0], pBf[0]))
    cIDy = int(max(pAi[1], pAf[1], pBi[1], pBf[1]))

    rotateImage = recImage[cSEx: cIDx, cSEy: cIDy]

    # rotateImage = imutils.rotate_bound(recImage, angulo)

    try:
        termo1 = math.acos((b / 2) / (d)) * (d ** 2 / math.sqrt(d ** 2 - (b / 2) ** 2))
        termo2 = math.acos((b / 2) / c) * (c ** 2 / math.sqrt(c ** 2 - (b / 2) ** 2))
        vFormulaArea = 2 * math.pi * (b / 2) ** 2 + math.pi * (b / 2) * (termo1 + termo2)
    except ValueError:
        vFormulaArea = 1
    try:
        vFormulaVolume = (b / 2) ** 2 * ((2 * math.pi) / 3) * (d + c)
    except ValueError:
        vFormulaVolume = 1

    # nomeArquivo = rf'{__path_folder}/Processed/{__root_file_name}_{nFile}.png'
    # rotateFileName = rf'{__path_folder}/Processed/rotate_{__root_file_name}_{nFile}.png'

    processedImageName = str(Path(processed_path, f'{nFile}.png'))
    rotateProcessedImageName = str(Path(processed_path, f'rotate_{nFile}.png'))

    leituraArquivo = f'{processedImageName},{a},{b},{c},{d},{v},{area},{vFormulaArea},{vFormulaVolume},{pixFactor}, {dFactor}\n'
    nFile += 1

    cv2.imwrite(str(processedImageName), imgProc)
    cv2.imwrite(str(rotateProcessedImageName), rotateImage)

    # imgProcess[linhas[x]:linhas[x + 1], colunas[y]:colunas[y + 1]] = imgProc
    imgProcess[lIni:lFin, cIni:cFin] = imgProc

    # Redimensiona para exibição
    tLines, tColumns, _ = adjustImageDimension(imgProcess)
    down_points = (tColumns, tLines)
    dispImage = cv2.resize(imgProcess, down_points, interpolation=cv2.INTER_LINEAR)

    # Desabilitar a exibição da imagem
    # cv2.imshow("Window", dispImage)
    # cv2.waitKey(1)

    return (leituraArquivo, lIni, lFin, cIni, cFin, imgProc)

# Esta função desenha linhas na imagem com base em coordenadas fornecidas para linhas
# de calibração, temporárias e de elementos.
# variaveis: image -> imagem na qual as linhas serão desenhadas
# calibrationLine -> coordenadas da linha de calibração
# temLines -> lista de coordenadas para linhas temp
# elementLines -> lista de coordenadas para linhas de elementos
# totalColumns -> numero total de colunas a srem desenhadas
# totalLines -> numero total de linhas a serem desenhadas
def drawLines(image, calibrationLine, tempLines, elementLines, totalColumns, totalLines):
    if len(calibrationLine) > 0:
        xi, yi, c = calibrationLine[0]
        xf, yf, c = calibrationLine[1]
        cv2.line(image, (xi, yi), (xf, yf), (0, 0, 255), 1)
    else:
        if len(tempLines) > 0:
            (posTemp, status) = tempLines[0]
            if status == 1:  # indica uma coluna
                cv2.line(image, (0, posTemp), (totalColumns - 1, posTemp), (0, 255, 0), 1)
            elif status == 2:
                cv2.line(image, (posTemp, 0), (posTemp, totalLines - 1), (255, 0, 0), 1)

        for i in elementLines:
            (posTemp, status) = i
            if status == 1:  # indica uma coluna
                cv2.line(image, (0, posTemp), (totalColumns - 1, posTemp), (0, 255, 0), 1)
            elif status == 2:
                cv2.line(image, (posTemp, 0), (posTemp, totalLines - 1), (255, 0, 0), 1)

    return (image)


# Essa função extrai sub-imagens de uma imagem maior com base
# em linhas temporárias e de elementos.
# variaveis: image -> imagem da qual as subimagens serão extraidas
# temLines -> lista de coordenadas para linhas temp
# elementLines -> lista de coordenadas para linhas de elementos
def subImagens(img, tColumns, tLines, factor, pixFactor, dFactor, nomeArquivo):
    global __path_file_name
    imgProcess = img.copy()
    results_folder_path = Path(os.getcwd(), 'results')
    # Create a new results directory because it does not exist
    check_and_create_directory_if_not_exist(results_folder_path)

    if nomeArquivo == '':
        new_file_name = dlg.asksaveasfilename(confirmoverwrite=False, initialdir=results_folder_path)
        __path_file_name = Path(new_file_name)
    else:
        __path_file_name = Path(nomeArquivo)

    # Check whether the specified path exists or not
    print('Check whether the specified path exists or not')
    file_path_results = Path(results_folder_path, __path_file_name.stem)
    check_and_create_directory_if_not_exist(file_path_results)

    # processed_path = Path(__path_file_name.parent, 'processed')
    processed_path = Path(file_path_results, 'processed')
    check_and_create_directory_if_not_exist(processed_path)

    # Check whether the specified path exists or not
    plot_results_path = Path(file_path_results, 'fit_plots_results')
    check_and_create_directory_if_not_exist(plot_results_path)

    # Create the report file path
    nomeRelatorio = Path(results_folder_path, 'Relatorio.dad')
    # Now creating the report file itself with its reference
    arqRelatorio = open(nomeRelatorio, 'at')

    posEggs = findeggs(img)
    lMin = cMin = +9999
    lMax = cMax = -9999

    img_bytes = imgProcess.tobytes()
    shape = imgProcess.shape

    # Limita numero de processos
    # multiprocessing.cpu_count() retorna a qnt disponivel de nucles logicos (threads de cpu) na maquina
    # o uso do -1 é para deixar ao menos 1 nucleo livre
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    # print(f"Executando com {max_workers} processos paralelos")
    # ProcessPoolExecutor "chama" a função process_single_egg paralelamente(isolados) executando em nucles diferentes da CPU
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        # Esse trecho envia todos os ovos para serem processados em paralelo usando múltiplos núcleos da CPU. Reduzindo o tempo de execução.
        for nFile, egg in enumerate(posEggs, 1):
            # Cada ovo detectado é processado por um "processo" separado (process_single_egg) por meio da linha abaixo
            futures.append(executor.submit(
                process_single_egg,
                egg, img_bytes, shape, factor, pixFactor, dFactor,
                str(processed_path), str(plot_results_path), nFile, str(__path_file_name)
            ))

        for future in futures:
            result = future.result()
            if result:
                leituraArquivo, lIni, lFin, cIni, cFin, imgProc = result
                arqRelatorio.write(leituraArquivo)
                imgProcess[lIni:lFin, cIni:cFin] = imgProc
                lMin = min(lMin, lIni)
                cMin = min(cMin, cIni)
                lMax = max(lMax, lFin)
                cMax = max(cMax, cFin)

    arqRelatorio.close()

    recImage = imgProcess[lMin:lMax, cMin:cMax]
    tLines, tColumns, _ = adjustImageDimension(recImage)
    down_points = (tColumns, tLines)
    dispImage = cv2.resize(recImage, down_points, interpolation=cv2.INTER_LINEAR)

    # Substituído: cv2.imshow() e waitKey() blocks
    resultado_img_path = Path(file_path_results, 'imagem_processada_final.png')
    cv2.imwrite(str(resultado_img_path), dispImage)
    print(f"Imagem final salva em: {resultado_img_path}")

def process(frame, factor, pixFactor, egg_num, path_file_name, egg_folder_fit_plot_path):

    # Measures that will be returned
    A = 0
    B = 0
    C = 0
    D = 0

    # Uma cópia da imagem original é feita para preservar a imagem na resolução original.
    original = frame.copy()
    # original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # A imagem é reduzida de acordo com um fator fornecido como argumento para otimizar o tempo de processamento.
    [lin, col, ch] = original.shape
    frame = resize(frame, [int(lin / factor), int(col / factor)])
    frame = np.uint8(frame * 255)

    # data = rgb2ycbcr(frame)
    #
    # # Define thresholds for channel 1 based on histogram settings
    # channel1Min = 123.000
    # channel1Max = 255.000
    #
    # # Define thresholds for channel 2 based on histogram settings
    # channel2Min = 0.000
    # channel2Max = 255.000
    #
    # # Define thresholds for channel 3 based on histogram settings
    # channel3Min = 0.000
    # channel3Max = 255.000

    data = rgb2hsv(frame)

    # % Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.000
    channel1Max = 1.000

    # % Define thresholds for channel 2 based on histogram settings
    channel2Min = 0.000
    channel2Max = 1.000

    # % Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.578
    channel3Max = 1.000

    # Creating the mask for segmentation
    print('Creating the mask for segmentation')
    data = np.bitwise_and(np.bitwise_and(np.bitwise_and(data[:, :, 0] >= channel1Min, data[:, :, 0] <= channel1Max),
                                         np.bitwise_and(data[:, :, 1] >= channel2Min, data[:, :, 1] <= channel2Max)),
                          np.bitwise_and(data[:, :, 2] >= channel3Min, data[:, :, 2] <= channel3Max))
    data[data is True] = 1
    data[data is False] = 0

    data = median(data, disk(3))
    data[data > 0] = 1

    # Process to identify the bounding box that contains the egg and subsequently improve segmentation
    print('Process to identify the bounding box that contains the egg and subsequently improve segmentation')
    [linoriginal, coloriginal] = data.shape
    labels = label(data)
    props = regionprops_table(labels, properties=('bbox', 'major_axis_length', 'minor_axis_length'))
    df = pd.DataFrame(props)

    fl_find_bb = False
    for index, row in df.iterrows():
        if 0.30 * linoriginal <= row['major_axis_length'] <= 1 * linoriginal and \
                0.15 * coloriginal <= row['minor_axis_length'] <= 1 * coloriginal:
            data = data.astype('uint8')
            data[int(row['bbox-0']):int(row['bbox-2']), int(row['bbox-1']):int(row['bbox-3'])] = \
                data[int(row['bbox-0']):int(row['bbox-2']), int(row['bbox-1']):int(row['bbox-3'])] + 1
            data[data == 1] = 0
            data[data == 2] = 255
            fl_find_bb = True
            break

    if not fl_find_bb:
        return [original, -1, -1, -1, -1]

    # Identifying the two points that form the longest straight line
    print('Identifying the two points that form the longest straight line')
    border_points = np.array(np.vstack(find_contours(data, 0.1)))
    index_size, _ = border_points.shape
    pt1 = pt2 = []
    max_distance = 0
    for i in range(1, index_size):
        for j in range(i + 1, index_size):
            distance = np.linalg.norm(border_points[i] * factor - border_points[j] * factor)
            if distance > max_distance:
                pt1 = border_points[i] * factor
                pt2 = border_points[j] * factor
                max_distance = distance

    # Draw the line of these points on the original image
    print('Draw the line of these points on the original image')
    if len(pt1) > 0 and len(pt2) > 0:
        original = cv2.line(original, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])), (47, 141, 255),
                            thickness=2)
        A = max_distance

    # Finds the angle of the line formed by the previous points and, later, finds the longest straight line
    print('Finds the angle of the line formed by the previous points and, later, finds the longest straight line')
    msRmaior = (pt1[1] * factor - pt2[1] * factor) / (pt1[0] * factor - pt2[0] * factor)
    ms = -1 / msRmaior
    ms = degrees(atan(ms))
    max_distance = 0
    pt3 = pt4 = []
    dicSlicesNoFit = {}
    xdata = []
    ydata = []

    for i in range(1, index_size):
        for j in range(1, index_size):
            if border_points[j][0] != border_points[i][0]:
                mdRmenor = (border_points[i][1] * factor - border_points[j][1] * factor) / (
                        border_points[i][0] * factor - border_points[j][0] * factor)
                bRmenor = border_points[i][1] - mdRmenor * border_points[i][0]  # defined constant of straight equation

                md = degrees(atan(mdRmenor))
                if 0 <= abs(md - ms) <= 0.3:
                    distance = np.linalg.norm(border_points[i] * factor - border_points[j] * factor)
                    distP1 = abs(-mdRmenor * pt1[0] + pt1[1] - bRmenor) / ((mdRmenor ** 2 + 1) ** 0.5)

                    original = cv2.line(original, (int(border_points[i][1]), int(border_points[i][0])),
                                        (int(border_points[j][1]), int(border_points[j][0])), (30, 105, 210),
                                        thickness=1)

                    dicSlicesNoFit[distP1] = distance
                    # !AQUI - Criar lista de X e Y para fazer o ajuste polinomial
                    xdata.append(distP1)
                    ydata.append(distance)

                    if distance > max_distance:
                        pt3 = border_points[i] * factor
                        pt4 = border_points[j] * factor

                        max_distance = distance

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    # List of polynomial curv functions to use
    print('List of polynomial curv functions to use')
    poly_degree_functions = [polynomial_curv_3, polynomial_curv_5, polynomial_curv_7, polynomial_curv_9,
                             polynomial_curv_11]

    # Calculate disc slices with curve fit and its errors
    dic_slices_fit, curve_fit_errors = disc_slices_curve_fit(xdata, ydata, poly_degree_functions, path_file_name, egg_num, egg_folder_fit_plot_path)


    # arqComparativo = open(rf'{__plot_results_path}/Relatorio_Comparativo_Volumes.dad', 'at')
    # arqComparativo = open(Path(egg_folder_fit_plot_path, 'Relatorio_Comparativo_Volumes.txt'), 'at')

    volume_results = {}

    # Measure egg volume based on trapezoidal rule
    print('Measure egg volume based on trapezoidal rule')
    (oldVolume, oldEggArea) = calcVolume(dicSlicesNoFit, pixFactor)
    volume_results["Antigo"] = (oldVolume, oldEggArea)

    # volume chosen to calculate the pixels at the final of this function
    chosen_volume = 0.0
    chosen_area = 0.0

    for poly_function in dic_slices_fit:
        print(f'Calculando e Escrevendo volume e área da função polinomial : {poly_function.__name__}')
        (poly_volume, poly_eggArea) = calcVolume(dic_slices_fit[poly_function], pixFactor)
        volume_results[poly_function] = (poly_volume, poly_eggArea)
        if poly_function == polynomial_curv_11:
            chosen_volume = poly_volume
            chosen_area = poly_eggArea

    """ arqComparativo.write(f'{__path_file_name.name} - {egg_name}\n\nVolume Antigo (sem ajuste polinomial de curva) = {oldVolume}\nArea antiga (sem ajuste de curva) = {oldEggArea}\n\n')
    last_volume = 0
    last_eggArea = 0
    for (poly_function, result_disc_slices) in dic_slices_fit:
        if(dic_slices_fit[-1] == (poly_function, result_disc_slices)).all():
            print(f'Calculando e Escrevendo Ultima function : {poly_function.__name__}')
            (last_volume, last_eggArea)=calcVolume(result_disc_slices, pixFactor)
            volume_results[poly_function] = (last_volume, last_eggArea)
            arqComparativo.write(f'Volume Novo ({poly_function.__name__}) = {last_volume}\nArea Nova ({poly_function.__name__}) = {last_eggArea}\n\n')
        else:
            print(f'Calculando e Escrevendo function : {poly_function.__name__}')
            (volume, eggArea)=calcVolume(result_disc_slices, pixFactor)
            volume_results[poly_function] = (last_volume, last_eggArea)
            arqComparativo.write(f'Volume Novo ({poly_function.__name__}) = {volume}\nArea Nova ({poly_function.__name__}) = {eggArea}\n\n')

    arqComparativo.close() """

    # Generate sheet file (.xlsx) with data of egg
    create_sheet(egg_folder_fit_plot_path, egg_num, volume_results, curve_fit_errors)

    # Draws the perpendicular line
    if len(pt1) > 0 and len(pt2) > 0:
        original = cv2.line(original, (int(pt3[1]), int(pt3[0])), (int(pt4[1]), int(pt4[0])), (0, 102, 0), thickness=2)
        B = max_distance

    # Finds the intersection of the lines and then draws the two sub-lines (Above and Below) of the perpendicular line
    inter = lineLineIntersection(pt1, pt2, pt3, pt4)
    if inter:
        distance_pt1_inter = np.linalg.norm(pt1 - inter)
        distance_pt2_inter = np.linalg.norm(pt2 - inter)
        if distance_pt1_inter > distance_pt2_inter:
            original = cv2.line(original, (int(pt1[1]), int(pt1[0])), (int(inter[1]), int(inter[0])), (0, 0, 255),
                                thickness=2)
            original = cv2.line(original, (int(pt2[1]), int(pt2[0])), (int(inter[1]), int(inter[0])), (255, 255, 255),
                                thickness=2)
            C = distance_pt1_inter
            D = distance_pt2_inter
        else:
            original = cv2.line(original, (int(pt2[1]), int(pt2[0])), (int(inter[1]), int(inter[0])), (0, 0, 255),
                                thickness=2)
            original = cv2.line(original, (int(pt1[1]), int(pt1[0])), (int(inter[1]), int(inter[0])), (255, 255, 255),
                                thickness=2)
            C = distance_pt2_inter
            D = distance_pt1_inter

    A *= pixFactor
    B *= pixFactor
    C *= pixFactor
    D *= pixFactor

    cv2.putText(original, 'A: ' + str("{:.3f}".format(A)), (original.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (47, 141, 255), 1)
    cv2.putText(original, 'B: ' + str("{:.3f}".format(B)), (original.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 102, 0), 1)
    cv2.putText(original, 'C: ' + str("{:.3f}".format(C)), (original.shape[1] - 150, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 255), 1)
    cv2.putText(original, 'D: ' + str("{:.3f}".format(D)), (original.shape[1] - 150, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 1)
    cv2.putText(original, 'V: ' + str("{:.3f}".format(chosen_volume)), (original.shape[1] - 150, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (30, 105, 210), 1)

    return [original, A, B, C, D, chosen_volume, chosen_area, pt1, pt2, pt3, pt4]


# Function to return the intersection between two lines
def lineLineIntersection(A, B, C, D):
    # Line 01
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1 * (A[0]) + b1 * (A[1])

    # Line 02
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2 * (C[0]) + b2 * (C[1])

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return False
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return [int(x), int(y)]


def calcVolume(slices, pixFactor):
    dicSlices = sorted(slices.items())
    volume = 0
    areaLateral = 0
    # print('Inicio Ovo')
    for t in range(1, len(dicSlices)):
        elem1 = dicSlices[t]
        elem0 = dicSlices[t - 1]
        h2, rMaior = elem1
        h1, rMenor = elem0

        # print ("(",h2,",",rMaior,")")

        h = abs(h2 - h1) * pixFactor
        rMaior = (rMaior / 2) * pixFactor
        rMenor = (rMenor / 2) * pixFactor
        g = math.sqrt(h ** 2 + (rMaior - rMenor) ** 2)
        volume += ((math.pi * h) / 3) * (rMaior ** 2 + rMenor ** 2 + rMaior * rMenor)
        areaLateral += math.pi * g * (rMaior + rMenor)
    volume = volume / 100
    return [volume, areaLateral]


def disc_slices_curve_fit(x_data, y_data, polynomial_fit_degree_functions, path_file_name, egg_num, egg_folder_fit_plot_path):
    # last_function = polynomial_fit_degree_functions[-1]
    resultDiscSlices = {}
    # resultCurveError = []
    resultCurveError = {}
    for poly_fit_function in polynomial_fit_degree_functions:
        # Criar a curva de ajuste polinomial do grau escolhido na lista
        popt, pcov = curve_fit(poly_fit_function, x_data, y_data)

        # Using 'lm' method for Levenberg-Marquardt algorithm or 'trf' method for Trust Region Reflective algorithm.
        # p0 = [10, 0.1, 1, 10, 0.1, 1, 1]
        # popt, pcov = curve_fit(poly_fit_function, x_data, y_data, p0=p0, method='lm')

        y_data_fit = poly_fit_function(x_data, *popt)

        # plotando o gráfico do ajuste com os os valores de coeficiente otimizados
        plt.cla()
        plt.plot(x_data, y_data, 'b-', label='Sem ajuste (antigo)')
        plt.plot(x_data, y_data_fit, 'r-', label=f'Com ajuste: {poly_fit_function.__name__}')
        plt.suptitle(f'{Path(path_file_name).name} - Ovo {egg_num}', fontweight='bold')
        plt.title(f'Comparação: Sem ajuste X C/ ajuste ({poly_fit_function.__name__})')
        plt.xlabel("Distância X")
        plt.ylabel("Pontos de distância Y")
        plt.legend()
        # plt.show()

        # plt.savefig(rf'{__plot_results_path}/{__root_file_name}_{poly_fit_function.__name__}')
        plt.savefig(Path(egg_folder_fit_plot_path, f'plot_{poly_fit_function.__name__}'))

        # erro do ajuste polinomial
        perr = np.sqrt(np.diag(pcov))
        print(f'Error of the curve fit - {poly_fit_function.__name__}: {perr}\n')

        # Convert lists (x_data, y_data_fit) to dictionary and append it to the result list as tuple
        resultDiscSlices[poly_fit_function] = dict(zip(x_data, y_data_fit))
        # resultCurveError.append((poly_fit_function, perr))
        resultCurveError[poly_fit_function] = perr

        # if(poly_fit_function == last_function):
        # using dict() and zip() to convert lists to dictionary
        # return dict(zip(x_data, y_data_fit))
    # print(f'FINAL RESULT : {resultDiscSlices}')
    # return [np.array(resultDiscSlices), resultCurveError]
    return [resultDiscSlices, resultCurveError]

# função detecta ovos em uma imagem com base em certos critérios de área e proporção.
# Ela retorna uma lista de ovos encontrados, onde cada ovo é representado por uma lista [grupo, linha, x, y, largura, altura]
def findeggs(originalImg):
    ovos = []
    (alt, larg, ch) = originalImg.shape
    AreaTotal = alt * larg
    # preprocess the image
    gray_img = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
    # Applying 7x7 Gaussian Blur
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    # Applying threshold
    threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(threshold, 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    # Loop through each component
    for i in range(1, totalLabels):
        # Area of the component
        area = values[i, cv2.CC_STAT_AREA]
        percArea = (area * 100) / AreaTotal
        aspectRatio = float(int(values[i, cv2.CC_STAT_HEIGHT]) / int(values[i, cv2.CC_STAT_WIDTH]))

        if (percArea > 0.4) and (percArea < 1) and (aspectRatio > 1.1) and (aspectRatio < 1.6):
            (col, lin) = centroid[i]
            x = values[i, cv2.CC_STAT_LEFT]
            y = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]
            elem = [0, lin, x, y, w, h]
            ovos.append(elem)
    posVet = 0
    grupo = 1
    # categoriza pela posição na linha, são classificados aquelas regiões cuja posição não linha variam abaixo de 10%
    for i in range(len(ovos)):
        if ovos[i][0] == 0:
            ovos[i][0] = grupo
            for k in range(i + 1, len(ovos)):
                if (abs(ovos[k][1] - ovos[i][1]) * 100) / ovos[i][1] < 10:
                    ovos[k][0] = grupo
            grupo += 1

    # ordena pela linha
    for i in range(0, len(ovos) - 1):
        for j in range(i + 1, len(ovos)):
            if ovos[j][0] < ovos[i][0]:
                troca = ovos[j]
                ovos[j] = ovos[i]
                ovos[i] = troca

    # ordena pela coluna
    # controle = 1
    for i in range(0, len(ovos) - 1):
        for j in range(i+1, len(ovos)):
            if ((ovos[j][0] == ovos[i][0]) and (ovos[j][2] < ovos[i][2])):
                troca = ovos[j]
                ovos[j] = ovos[i]
                ovos[i] = troca
    return ovos

"""
Representação da curva - Curva polinomial de grau 3
A função deve receber as coordenadas X e Y de seus pontos de dados como entradas e retornar os valores Y previstos para cada valor X.
"""


def polynomial_curv_3(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def polynomial_curv_5(x, a, b, c, d, e, f):
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f


def polynomial_curv_7(x, a, b, c, d, e, f, g, h):
    return a * x ** 7 + b * x ** 6 + c * x ** 5 + d * x ** 4 + e * x ** 3 + f * x ** 2 + g * x + h


def polynomial_curv_9(x, a, b, c, d, e, f, g, h, i, j):
    return a * x ** 9 + b * x ** 8 + c * x ** 7 + d * x ** 6 + e * x ** 5 + f * x ** 4 + g * x ** 3 + h * x ** 2 + i * x + j


def polynomial_curv_11(x, a, b, c, d, e, f, g, h, i, j, k, m):
    return a * x ** 11 + b * x ** 10 + c * x ** 9 + d * x ** 8 + e * x ** 7 + f * x ** 6 + g * x ** 5 + h * x ** 4 + i * x ** 3 + j * x ** 2 + k * x + m


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x <= x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])


# def sigmoid(x, a, b, c, d):
# y = a / (1 + np.exp(-b*(x-c))) + d
# return y

# Não funcionou : Linha indo para a direita, sem seguir os pontos
def sigmoid(x, a, b, c, d):
    z = b * (x - c)
    z = np.clip(z, -500, 500)  # limit the values of the exponential term
    return a / (1 + np.exp(-z)) + d


# Resultado: Semelhante a regressão polinomial de grau 3
def gaussian(x, a, b, c, d):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d


def get_polynomial_curv_portuguese_name(polynomial_func):
    if polynomial_func == polynomial_curv_3:
        return "Ajuste Polinomial Grau 3"
    elif polynomial_func == polynomial_curv_5:
        return "Ajuste Polinomial Grau 5"
    elif polynomial_func == polynomial_curv_7:
        return "Ajuste Polinomial Grau 7"
    elif polynomial_func == polynomial_curv_9:
        return "Ajuste Polinomial Grau 9"
    elif polynomial_func == polynomial_curv_11:
        return "Ajuste Polinomial Grau 11"
    else:
        return "Ajuste desconhecido"


# This function consists of the sum of three exponential functions with different amplitudes,
# centers, and widths, plus a constant d. You can adjust the initial parameter values to get a better fit.
# def exponential_combo(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, d):
# return a1 * np.exp(-((x - b1)/c1)**2) + a2 * np.exp(-((x - b2)/c2)**2) + a3 * np.exp(-((x - b3)/c3)**2) + d
# Esta função cria ou atualiza uma planilha Excel com os resultados do volume e os erros
# de ajuste para diferentes tipos de ajuste polinomial
def create_sheet(path_file_name, egg_name, volume_results, curve_fit_errors):
    ordem_valores = ["Antigo", polynomial_curv_3, polynomial_curv_5, polynomial_curv_7, polynomial_curv_9,
                     polynomial_curv_11]

    if isinstance(path_file_name, Path):
        sheet_name_directory = Path(path_file_name.parents[2], 'Comparacao_Volume_Area.xlsx')
        file_name = path_file_name.parents[1].stem

        if os.path.exists(sheet_name_directory):
            try:
                workbook = load_workbook(sheet_name_directory)
            except Exception as e:
                print(f"Arquivo '{sheet_name_directory}' corrompido ou inválido. Criando novo workbook. Erro: {e}")
                workbook = Workbook()
        else:
            workbook = Workbook()

        sheetnames = workbook.sheetnames
        if file_name not in sheetnames:
            workbook.create_sheet(file_name)
            file_worksheet = workbook[file_name]
            file_worksheet["B1"] = "Volume - ANTIGO"
            file_worksheet["C1"] = "Area - ANTIGO"
            file_worksheet["D1"] = "Volume - POLINOM. GRAU 3"
            file_worksheet["E1"] = "Area - POLINOM. GRAU 3"
            file_worksheet["F1"] = "Erros - POLINOM. GRAU 3"
            file_worksheet["G1"] = "Volume - POLINOM. GRAU 5"
            file_worksheet["H1"] = "Area - POLINOM. GRAU 5"
            file_worksheet["I1"] = "Erros - POLINOM. GRAU 5"
            file_worksheet["J1"] = "Volume - POLINOM. GRAU 7"
            file_worksheet["K1"] = "Area - POLINOM. GRAU 7"
            file_worksheet["L1"] = "Erros - POLINOM. GRAU 7"
            file_worksheet["M1"] = "Volume - POLINOM. GRAU 9"
            file_worksheet["N1"] = "Area - POLINOM. GRAU 9"
            file_worksheet["O1"] = "Erros - POLINOM. GRAU 9"
            file_worksheet["P1"] = "Volume - POLINOM. GRAU 11"
            file_worksheet["Q1"] = "Area - POLINOM. GRAU 11"
            file_worksheet["R1"] = "Erros - POLINOM. GRAU 11"
        else:
            file_worksheet = workbook[file_name]

        # Find / Create egg row
        # is_new_egg = False
        egg_row = 0
        for row in file_worksheet.iter_rows(min_col=1, max_col=1):
            for cell in row:
                if cell.value == egg_name:
                    egg_row = cell.row
        if egg_row == 0:
            max_col_row = len([cell for cell in file_worksheet["A"] if cell.value])
            egg_row = max_col_row + 2
            file_worksheet[f"A{egg_row}"] = egg_name
            # is_new_egg = True

        current_min_col = 2
        for value_type in ordem_valores:
            (volume, area) = volume_results[value_type]
            file_worksheet.cell(row=egg_row, column=current_min_col).value = volume
            file_worksheet.cell(row=egg_row, column=current_min_col + 1).value = area
            if value_type == "Antigo":
                current_min_col = current_min_col + 2
            else:
                file_worksheet.cell(row=egg_row, column=current_min_col + 2).value = ', '.join(
                    str(error) for error in curve_fit_errors[value_type])
                current_min_col = current_min_col + 3

        workbook.save(sheet_name_directory)
    else:
        print(f"input should be of type Path (pathlib) : {path_file_name}")


def exponential_combo(x, a1, b1, c1, a2, b2, c2, d):
    y1 = a1 * np.exp(-b1 * x) + c1
    y2 = a2 * np.exp(-b2 * (x - c2) ** 2) + d
    return y1 + y2

