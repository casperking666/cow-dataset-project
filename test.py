from ultralytics import YOLO


# Load the trained model
# model = YOLO('./models/yolov8s-cow-crops-a100-20e.pt') 
model = YOLO('./runs/classify/train30/weights/best.pt')

# Define the test dataset path
# test_data_path = "/user/work/yf20630/cow-dataset-project/datasets/cow_cls"

# # Evaluate the model on the test set
# results = model.val(data=test_data_path, split="test", imgsz=640)
# print(results)

# # Optionally, print the top-1 and top-5 accuracy
# top1_accuracy = results.top1
# top5_accuracy = results.top5
# print(f"Top-1 Accuracy: {top1_accuracy}")
# print(f"Top-5 Accuracy: {top5_accuracy}")


# Define path to the image file
source = "/user/work/yf20630/cow-dataset-project/datasets/cow_cls/test/class1/pmfeed_4_3_16_frame_1802_cow_1.jpg"

# Run inference on the source
results = model(source)  # list of Results objects