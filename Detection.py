import cv2
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import torch

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
ig = Image.open("content/desktop-wallpaper-new-york-city-street-high.jpg")
transform = T.ToTensor()
img = transform(ig)
with torch.no_grad():
  pred = model([img])
  
coco_names = ["person" , "bicycle" , "car" , "motorcycle" , "airplane" , "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" , "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "hat" , "backpack" , "umbrella" , "shoe" , "eye glasses" , "handbag" , "tie" , "suitcase" ,
"frisbee" ,"numberplate", "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" ,
"baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" ,
"plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" ,
"banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" ,
"pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,
"mirror" , "dining table" , "window" , "desk" , "toilet" , "door" , "tv" ,
"laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
"oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
"clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"]

pred[0].keys()
bboxes,labels,score = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
num = torch.argwhere(score>0.8).shape[0]

font = cv2.FONT_HERSHEY_SIMPLEX
igg = cv2.imread("content/desktop-wallpaper-new-york-city-street-high.jpg")
for i in range(num):
    x1,y1,x2,y2 = bboxes[i].numpy().astype("int")
    class_name = coco_names[labels.numpy()[i]-1]
    igg = cv2.rectangle(igg,(x1,y1),(x2,y2),(0,255,0),1)
    igg = cv2.putText(igg,class_name,(x1,y1-10),font,0.5,(255,0,0),1,cv2.LINE_AA)

cv2.imshow('Image', igg)
cv2.waitKey(0)
cv2.destroyAllWindows()
