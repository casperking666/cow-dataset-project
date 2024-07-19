import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import DefaultDataCollator
import os
import numpy as np

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Define image transformations
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
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

dataset = dataset.map(preprocess_images, batched=True)

# Remove unused columns
dataset = dataset.remove_columns(['image'])

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
    evaluation_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=40,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs_vit',
    logging_steps=10,
)

# Create a data collator
data_collator = DefaultDataCollator()

# Load metrics
accuracy_metric = load_metric('accuracy', trust_remote_code=True)
f1_metric = load_metric('f1', trust_remote_code=True)

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
    tokenizer=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)
