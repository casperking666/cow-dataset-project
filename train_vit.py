import torch
from transformers import AutoImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import evaluate
import numpy as np

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the image processor
image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Define image transformations
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    normalize,
])

# Load dataset
dataset = load_dataset('/user/work/yf20630/cow-dataset-project/datasets', data_dir='subset_tiny')

# Preprocess the dataset
def preprocess_images(examples):
    examples['pixel_values'] = [transforms(image.convert('RGB')) for image in examples['image']]
    return examples

dataset = dataset.map(preprocess_images, batched=True, remove_columns=['image'])

# Split into train and validation sets
train_dataset = dataset['train']
val_dataset = dataset['test']

# Load the model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224', 
    num_labels=len(dataset['train'].features['label'].names),
    ignore_mismatched_sizes=True
)
model.to(device)  # Move the model to the GPU if available

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results_vit',
    eval_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=40,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs_vit',
    logging_steps=10,
)

# Load metrics
accuracy_metric = evaluate.load('accuracy')
f1_metric = evaluate.load('f1')

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)
    f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average='weighted')
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)