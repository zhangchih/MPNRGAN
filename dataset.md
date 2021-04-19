# Prepare the dataset
The overall layout of dataset folder ```root/datasets``` is as
```
├── Datasets
    ├── TrainingSet
    ├── PriorsSet
    ├── TestSet
```
In this implementation, we organized the data in the form of H5 files. The specific organization form of H5 file is
```
├── H5 File Name   
    ├── Block ID1                   
        ├── data                  
        ├── label
    ├── Block ID2                   
        ├── data                  
        ├── label
    ...
    ├── Block IDN                   
        ├── data                  
        ├── label
```
Due to the huge amount of data, we split the entire data set into multiple H5 files. 

The dataset folder path is ```root/dataset_folder```.

If you want to train this on your dataset, you can rewrite the dataset.py file to adopt to your data.

Both VISoR-40 and BigNeuron dataset are public datasets. You can download them refer to [VISoR-40](https://braindata.bitahub.com/) and [BigNeuron](https://alleninstitute.org/bigneuron/about/).
