import argparse
import torch
from transformers import AutoImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np
from transformers.integrations import TensorBoardCallback
import os
import random
from torchvision.transforms import v2

def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT model with augmentation parameters")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--run_number', type=str, default="17")
    parser.add_argument('--dataset_path', type=str, default='/user/work/yf20630/cow-dataset-project/datasets')
    parser.add_argument('--seed', type=int, default=42)
    
    # Add augmentation parameters, all defaulting to None
    parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--random_resized_crop', type=float, nargs=2, default=None)
    parser.add_argument('--random_resize', type=int, nargs=2, default=None)
    parser.add_argument('--zoom_out', type=float, nargs=2, default=None)
    parser.add_argument('--rotation_degrees', type=float, default=None)
    parser.add_argument('--affine', type=float, nargs=4, default=None)  # degrees, translate, scale_min, scale_max
    parser.add_argument('--perspective', type=float, nargs=2, default=None)  # distortion_scale, p
    parser.add_argument('--color_jitter', type=float, nargs=4, default=None)  # brightness, contrast, saturation, hue
    parser.add_argument('--grayscale_p', type=float, default=None)
    parser.add_argument('--gaussian_blur', type=float, nargs=2, default=None)  # kernel_size, sigma
    parser.add_argument('--gaussian_noise_std', type=float, default=None)
    parser.add_argument('--posterize_bits', type=int, default=None)
    parser.add_argument('--solarize_threshold', type=int, default=None)
    parser.add_argument('--sharpness_factor', type=float, default=None)
    parser.add_argument('--autocontrast_p', type=float, default=None)
    parser.add_argument('--equalize_p', type=float, default=None)
    parser.add_argument('--auto_augment', type=str, default=None, choices=['imagenet', 'randaugment', 'trivialaugmentwide'])
    
    return parser.parse_args()

def get_transforms(args, image_processor):
    normalize = v2.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    
    train_transforms = []
    
    if args.resize:
        train_transforms.append(v2.Resize(size=(args.resize, args.resize)))
    
    if args.random_resized_crop:
        train_transforms.append(v2.RandomResizedCrop(size=(224, 224), scale=args.random_resized_crop))
    
    if args.random_resize:
        train_transforms.append(v2.RandomResize(min_size=args.random_resize[0], max_size=args.random_resize[1]))
    
    if args.zoom_out:
        train_transforms.append(v2.RandomZoomOut(fill=0, side_range=args.zoom_out))
    
    if args.rotation_degrees:
        train_transforms.append(v2.RandomRotation(degrees=args.rotation_degrees))
    
    if args.affine:
        train_transforms.append(v2.RandomAffine(degrees=args.affine[0], translate=(args.affine[1], args.affine[1]), 
                                                scale=(args.affine[2], args.affine[3])))
    
    if args.perspective:
        train_transforms.append(v2.RandomPerspective(distortion_scale=args.perspective[0], p=args.perspective[1]))
    
    if args.color_jitter:
        train_transforms.append(v2.ColorJitter(brightness=args.color_jitter[0], contrast=args.color_jitter[1], 
                                               saturation=args.color_jitter[2], hue=args.color_jitter[3]))
    
    if args.grayscale_p:
        train_transforms.append(v2.RandomGrayscale(p=args.grayscale_p))
    
    if args.gaussian_blur:
        train_transforms.append(v2.GaussianBlur(kernel_size=int(args.gaussian_blur[0]), sigma=args.gaussian_blur[1]))
    
    if args.gaussian_noise_std:
        train_transforms.append(v2.Lambda(lambda x: x + torch.randn_like(x) * args.gaussian_noise_std))
    
    if args.posterize_bits:
        train_transforms.append(v2.RandomPosterize(bits=args.posterize_bits))
    
    if args.solarize_threshold:
        train_transforms.append(v2.RandomSolarize(threshold=args.solarize_threshold))
    
    if args.sharpness_factor:
        train_transforms.append(v2.RandomAdjustSharpness(sharpness_factor=args.sharpness_factor))
    
    if args.autocontrast_p:
        train_transforms.append(v2.RandomAutocontrast(p=args.autocontrast_p))
    
    if args.equalize_p:
        train_transforms.append(v2.RandomEqualize(p=args.equalize_p))
    
    if args.auto_augment:
        if args.auto_augment == 'imagenet':
            train_transforms.append(v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET))
        elif args.auto_augment == 'randaugment':
            train_transforms.append(v2.RandAugment(num_ops=2, magnitude=9))
        elif args.auto_augment == 'trivialaugmentwide':
            train_transforms.append(v2.TrivialAugmentWide())
    
    train_transforms.extend([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        normalize,
    ])
    
    train_transforms = v2.Compose(train_transforms)
    print(train_transforms)
    
    val_transforms = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        normalize,
    ])
    
    return train_transforms, val_transforms

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_val(examples, val_transforms):
    examples["pixel_values"] = [val_transforms(image.convert("RGB")) for image in examples["image"]]
    return examples

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def preprocess_train(example_batch, train_transforms):
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch['image']
    ]
    return example_batch

def load_and_preprocess_dataset(dataset_path, train_transforms, val_transforms):
    dataset = load_dataset(dataset_path, data_dir='subset_tiny')
    train = dataset["train"]
    val = dataset["test"]

    val.set_transform(lambda examples: preprocess_val(examples, val_transforms))
    train.set_transform(lambda examples: preprocess_train(examples, train_transforms))
    return train, val

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
        remove_unused_columns=False,
        num_train_epochs=num_epochs,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=log_dir,
        logging_steps=10,
        report_to=["tensorboard"],
        seed=seed,
        dataloader_drop_last=True,
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
        data_collator=collate_fn,
        callbacks=[TensorBoardCallback()]
    )
    
    trainer.train()
    results = trainer.evaluate()
    return results

def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    train_transforms, val_transforms = get_transforms(args, image_processor)
    
    train_dataset, val_dataset = load_and_preprocess_dataset(args.dataset_path, train_transforms, val_transforms)
    
    model = load_model(num_labels=len(train_dataset.features['label'].names), device=device)
    
    log_dir = get_log_dir(args.batch_size, args.num_epochs, args.run_number, args.seed)
    os.makedirs(log_dir, exist_ok=True)
    
    training_args = get_training_args(args.batch_size, args.num_epochs, log_dir, args.seed)
    
    results = train_and_evaluate(model, train_dataset, val_dataset, training_args)
    print(results)

if __name__ == "__main__":
    main()