import comet_ml
from ultralytics import YOLO


comet_ml.init(project_name="cow-project-subset-small")


model = YOLO("yolov8s-cls.yaml")  # build a new model from YAML
# model = YOLO('yolov8s-cls.pt')  # Load model
# model = YOLO('runs/classify/train23/weights/last.pt')  # Load model

# results = model.train(
#     data = "/user/work/yf20630/cow-dataset-project/datasets/subset_small", 
#     epochs=400, 
#     imgsz=640, 
#     device=0, 
#     patience=20,
#     save_period=1,
#     hsv_h=0.02,
#     hsv_s=0.7,
#     hsv_v=0.4,
#     degrees=0.0,
#     translate=0.0,
#     scale=0.0,
#     shear=0.0,
#     perspective=0.0,
#     flipud=0.0,
#     fliplr=0.0,
#     mosaic=0.0,
#     mixup=0.0,
#     copy_paste=0.0
# )

results = model.train(
    data = "/user/work/yf20630/cow-dataset-project/datasets/subset_small", 
    epochs=150, 
    imgsz=640, 
    device=0, 
    save_period=1,
    scale=0.0,
    fliplr=0.0
)


# Evaluate the model
metrics = model.val(data = "/user/work/yf20630/cow-dataset-project/datasets/subset_small", split="test")
print(metrics)



# Inference
# results = model('/user/work/yf20630/cow-dataset-project/datasets/cow_cls/test')  # Update to the correct path
# results.show()
