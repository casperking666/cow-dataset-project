import torch
from transformers import AutoImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import evaluate
import numpy as np
from transformers.integrations import TensorBoardCallback
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms(image_processor):
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    return Compose([
        Resize((224, 224)),
        ToTensor(),
        normalize,
    ])

def preprocess_images(examples, transforms):
    examples['pixel_values'] = [transforms(image.convert('RGB')) for image in examples['image']]
    return examples

def load_and_preprocess_dataset(dataset_path, transforms):
    dataset = load_dataset(dataset_path, data_dir='subset_tiny_robo_aug')
    dataset = dataset.map(lambda examples: preprocess_images(examples, transforms), batched=True, remove_columns=['image'])
    return dataset['train'], dataset['test']

def load_model(num_labels, device):
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224', 
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    return model.to(device)

def get_log_dir(batch_size, num_epochs, run_number, seed):
    return f'./logs_vit/bs={batch_size}_e={num_epochs}_run_{run_number}_seed_{seed}'

def get_training_args(batch_size, num_epochs, log_dir, seed):
    return TrainingArguments(
        output_dir='./results_vit',
        eval_strategy='epoch',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=log_dir,
        logging_steps=10,
        report_to=["tensorboard"],
        seed=seed,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
    )

def compute_metrics(p):
    accuracy_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')
    
    preds = np.argmax(p.predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)
    f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average='weighted')
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

def train_and_evaluate(model, train_dataset, val_dataset, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[TensorBoardCallback()]
    )
    
    trainer.train()
    results = trainer.evaluate()
    return results

def main():
    # Set your parameters
    batch_size = 8
    num_epochs = 40
    run_number = 6
    dataset_path = '/user/work/yf20630/cow-dataset-project/datasets'
    seed = 42

    device = get_device()
    image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    transforms = get_transforms(image_processor)
    
    train_dataset, val_dataset = load_and_preprocess_dataset(dataset_path, transforms)
    
    model = load_model(num_labels=len(train_dataset.features['label'].names), device=device)
    
    log_dir = get_log_dir(batch_size, num_epochs, run_number, seed)
    os.makedirs(log_dir, exist_ok=True)
    
    training_args = get_training_args(batch_size, num_epochs, log_dir, seed)
    
    results = train_and_evaluate(model, train_dataset, val_dataset, training_args)
    print(results)

if __name__ == "__main__":
    main()