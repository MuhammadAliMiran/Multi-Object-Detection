import gradio as gr
import cv2
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch

def detect_objects(image):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    transform = T.ToTensor()
    img = transform(image)
    with torch.no_grad():
        pred = model([img])

    coco_names = ["person" , "bicycle" , "car" , "motorcycle" , "airplane" , "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" , "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "hat" , "backpack" , "umbrella" , "shoe" , "eye glasses" , "handbag" , "tie" , "suitcase" ,
    "frisbee" ,"numberplate", "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" ,
    "baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle",
    "plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" ,
    "banana" , "apple" , "sandwich", "orange" , "broccoli", "carrot" , "hot dog" ,
    "pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,
    "mirror" , "dining table" , "window" , "desk" , "toilet" , "door" , "tv" ,
    "laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
    "oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
    "clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"]

    pred[0].keys()
    bboxes, labels, score = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
    num = torch.argwhere(score > 0.8).shape[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = np.array(image)
    img_h, img_w, _ = img.shape
    font_scale = max(img_h, img_w) / 1000.0  # Adjust the divisor as needed for appropriate scaling
    font_thickness = max(1, int(font_scale))  # Ensure minimum thickness of 1
    
    for i in range(num):
        x1, y1, x2, y2 = bboxes[i].numpy().astype("int")
        class_name = coco_names[labels.numpy()[i] - 1]
        text_size = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        img = cv2.putText(img, class_name, (x1, y1 - 10), font, font_scale, (255, 255, 255), font_thickness + 2, cv2.LINE_AA, False)


    return img

app = gr.Interface(fn=detect_objects, inputs="image", outputs="image")
app.launch()
