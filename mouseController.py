import cv2

# function which will be called on mouse input
def drawLines(action, x, y, flags, *userdata):
    # Referencing global variables
    global elementLines, tempLines, originalImage, clicked, totalLines, totalColumns
    # Mark the top left corner when left mouse button is pressed

    if action == cv2.EVENT_LBUTTONDOWN:
        clicked = 1;
        tempLines = [(y, clicked)]
        # When left mouse button is released, mark bottom right corner
    elif action == cv2.EVENT_LBUTTONUP:
        elementLines.append((y, clicked))
        clicked = 0;
        # Draw the rectangle
    elif action == cv2.EVENT_RBUTTONDOWN:
        clicked = 2;
        tempLines = [(x, clicked)]
    elif action == cv2.EVENT_RBUTTONUP:
        elementLines.append((x,clicked))
        clicked = 0;
    elif action == cv2.EVENT_MOUSEMOVE:
        if clicked != 0:
            tempLines = []
            if clicked == 1:
                tempLines = [(y,clicked)]
            if clicked == 2:
                tempLines = [(x, clicked)]


    image=originalImage.copy()
    if clicked != 0:
        (posTemp, status) = tempLines[0]
        if status == 1: #indica uma coluna
            cv2.rectangle(image, (0,posTemp), (totalColumns-1,posTemp),(0,255,0),2)
        elif status == 2:
            cv2.rectangle(image, (posTemp,0), (posTemp,totalLines-1), (255, 0, 0), 2)

    for i in elementLines:
        (posTemp, status) = i
        if status == 1:  # indica uma coluna
            cv2.rectangle(image, (0, posTemp), (totalColumns - 1, posTemp), (0, 255, 0), 2)
        elif status == 2:
            cv2.rectangle(image, (posTemp, 0), (posTemp, totalLines - 1), (255, 0, 0), 2)

    cv2.imshow("Window", image)
