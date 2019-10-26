from fastai.vision import *
import cv2
import torch
import pickle
from pathlib import Path

path = Path("/Users/davidabraham/gesture-lingua-backend/experiments/notebooks")

with open(path / "class_names.pkl", "rb") as pkl_file:
    classes = pickle.load(pkl_file)


data2 = ImageDataBunch.single_from_classes(
    path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = cnn_learner(data2, models.resnet34)
learn.load('model-after-unfreeze')

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()  # capture each frame
    #     ret = cap.set(3,224)
    #     ret = cap.set(4,224)

    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   #converts each frame to gray

    cv2.imshow('FRAME', frame)
    # cv2.imshow('GRAY',gray)

    # For capturing frame and saving it as an image at given folder:
    if cv2.waitKey(1) == ord('n'):
        cv2.imwrite('test1.jpg', frame)
        img = open_image(Path('./test1.jpg'))
        label, index, pred = learn.predict(img)
        cv2.putText(frame, "Alphabet = {}".format(label), (380, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 215), 2)
        cv2.putText(frame, "Prob = {0:.4f}".format(torch.max(pred).item(
        )), (380, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Predictions", frame)
        print("Current alphabet =\n", torch.max(pred).item())
        print("\t\t " + str(label) + "\t\t")

    # making actual prediction for each frame:=============================================
    # For quitting the given session: ord is used to obtain unicode of the given string.
    # cv2.waitKey returns the unicode of the key which is pressed

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
