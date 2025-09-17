import cv2

image_path="./Screw_2.png"

image=cv2.imread(image_path)

FIRST_NUT_HEIGHT=108
MIDDLE_NUTS_HEIGHT=52

MIDDLE_NUT_START=46
MIDDLE_NUT_END=122

COLORS_DETECTED=[]

first_nut_subimage=image[FIRST_NUT_HEIGHT-MIDDLE_NUTS_HEIGHT:FIRST_NUT_HEIGHT, MIDDLE_NUT_START:MIDDLE_NUT_END]
color=first_nut_subimage.mean()
COLORS_DETECTED.append(color)
cv2.imshow(f"first nut {color=}", first_nut_subimage)

key = cv2.waitKey(0) & 0xFF
if key == ord('q'):
    exit()

for k in range(3):
    nut_subimage=image[FIRST_NUT_HEIGHT+MIDDLE_NUTS_HEIGHT*k: FIRST_NUT_HEIGHT+MIDDLE_NUTS_HEIGHT*(k+1), MIDDLE_NUT_START:MIDDLE_NUT_END]
    
    color=nut_subimage.mean()
    COLORS_DETECTED.append(color)
    cv2.imshow(f"{k} middle nut {color=}", nut_subimage)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break


