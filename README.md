# cow-dataset-project

This repo is pretty messy tbh, it contains a lot log files and images that might not be super relevant. But the important files are `create_subset.py` for creating different sizes of subsets. `train_vit_randAug.py` and train.py are for training vit and yolo respectively, the other versions of the train script are there as some pre-attempt. To run experiments that generate the results shown in the thesis (basically the pipeline referred in the thesis).

run `augmentation_search_yolo_general.py` or `augmentation_search_vit.py` for yolo or vit respectively. Under the hood, this will call the yolo cli or train_vit_parser.py for running the actual training. After which there will be a bunch log files generated in the specific folders, `parse_logs_yolo.py` and `parse_log_vit.py` are there for you to parse all logs in the directory to a single csv containing the summary of all runs. Finally run `vis_aug_res_vit.py` for generating the pretty plots shown in the thesis, note this works for both yolo and vit (the file name is misleading, it really should be called general or sth).

`create_augment_dateset.py` is for generating augmented dataset for the offline augmentation part.
