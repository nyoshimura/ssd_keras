# Work In Progress
# Now just reading usbcam... 2018/7/10

import cv2
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
import random

# Set the image size.
img_height = 300
img_width = 300

# 1: Build the Keras model
K.clear_session() # Clear previous models from memory.
model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)
# 2: Load the trained weights into the model.
weights_path = '../Model/SSD-keras/VGG_VOC0712_SSD_300x300_iter_120000.h5'
model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

# Set the colors for the bounding boxes
colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for i in range(len(classes))]

def process_image(img):
    # pre-processing
    resized = cv2.resize(img, (img_height, img_width))
    expanded = np.expand_dims(resized, 0)
    # predict
    y_pred = model.predict(expanded)
    # filter results
    confidence_threshold = 0.5
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold]
                     for k in range(y_pred.shape[0])]

    # calcurate bounding boxes
    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = int(box[2] * img.shape[1] / img_width)
        ymin = int(box[3] * img.shape[0] / img_height)
        xmax = int(box[4] * img.shape[1] / img_width)
        ymax = int(box[5] * img.shape[0] / img_height)
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)
        cv2.putText(img, label, (xmin, ymin+25), cv2.FONT_HERSHEY_PLAIN, 2, color, 5)
    result = img
    return result
####################################################

# capture usb cam
cap = cv2.VideoCapture(0)

while True:
    # read 1 frame from VideoCapture
    ret, frame = cap.read()
    # show raw image
    #cv2.imshow('Raw Frame', frame)
    # main process
    appliedSSD = process_image(frame)
    cv2.imshow('Result', appliedSSD)
    # wait 1ms for key input & break if k=27(esc)
    k = cv2.waitKey(1)
    if k==27:
        break

# release capture & close window
cap.release()
cv2.destroyAllWindows()
