from ultralytics import YOLO
from ultralytics import YOLOE
from multiprocessing import freeze_support

def main():

    # Load a COCO-pretrained YOLO26n model
    model = YOLO("yolo26n.pt")

    # Train the model on the COCO8 example dataset for 100 epochs
    #results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

    # Run inference with the YOLO26n model on the 'bus.jpg' image
    results = model("C:/Users/CHAMA COMPUTERS/Desktop/Data_Science/AI_ML/projects/YOLO26_demo/YOLO26_Demo/app/images/image1.jpeg")
    print

if __name__ == "__main__":
    freeze_support()   # Safe on Windows
    main() 