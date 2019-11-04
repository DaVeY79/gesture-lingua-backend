from fastai.vision import (
    ImageDataBunch,
    get_transforms,
    cnn_learner,
    imagenet_stats,
    open_image,
    models,
)
import numpy as np
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

c = 0

cap = cv2.VideoCapture(0)

res, score = '', 0.0
i = 0
mem = ''
consecutive = 0
sequence = ''

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    if ret:
        x1, y1, x2, y2 = 600, 50, 1200, 650
        img_cropped = img[y1:y2, x1:x2]

        c += 1
        # image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
        cv2.imwrite('test1.jpg', img_cropped)

        a = cv2.waitKey(1)  # waits to see if `esc` is pressed

        if i == 4:
            img_ = open_image(Path('./test1.jpg'))
            label, index, pred = learn.predict(img_)
            res = str(label)
            score = pred[0]

            i = 0
            if mem == res:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive == 2 and res not in ['nothing']:
                if res == 'space':
                    sequence += ' '
                elif res == 'del':
                    sequence = sequence[:-1]
                else:
                    sequence += res
                consecutive = 0
        i += 1
        cv2.putText(img, '%s' % (res.upper()), (100, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4)
        # cv2.putText(img, '(score = %.5f)' % (float(score)),
        cv2.putText(img, '(score = {})'.format(float(score)),
                    (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        mem = res
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow("img", img)
        img_sequence = np.zeros((200, 1200, 3), np.uint8)
        cv2.putText(img_sequence, '%s' % (sequence.upper()),
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('sequence', img_sequence)
        # resize_img_sequence = cv2.resize(img_sequence,(img_sequence.shape[0],img.shape[1]))
        # img = np.vstack((img,img_sequence))

        if (cv2.getWindowProperty('img', 0) == -1) or (a == 27):  # when `esc` is pressed
            break

# Following line should... <-- This should work fine now
cv2.destroyAllWindows()
cap.release()
