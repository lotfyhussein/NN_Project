# Neural Netwroks Project

This is our implementation for NNTI Project WS-2020

## Running Part_1
We provide a notebook that contains our implementation with outputs of latest run in each cell.
Also you can download the [checkpoint](https://drive.google.com/file/d/11kvdiP5GDMsLwHeLp4IzD2C4UNUWBir5/view?usp=sharing)  and load it directly. 


### Running Part_2

1- Download the  [checkpoint](https://drive.google.com/file/d/1SAStVRRHkWMRH230UifAc7uuzF6zakns/view?usp=sharing)


2- Please follow this folder structure:

    ├── leftImg8bit  # CityScapes Images
    ├── gtFine       # CityScapes Labels
    ├── Part_2         
    │   ├── train.py         
    │   ├── eval.py     
    │   ├── dataset.py           
    │   ├── network.py  
    │   ├── model-R2U-Net.cpt43     # Checkpoint file
    └── ...

To train the model
```
python3 train.py
```

To test the model
```
python3 eval.py
```


### Running Part_3

1- Download the  [checkpoint](https://drive.google.com/file/d/1PjoKCiy5j569EM0jxl0IP_NTr4Ayx94K/view?usp=sharing)

2- Please follow this folder structure:

    ├── leftImg8bit  # CityScapes Images
    ├── gtFine       # CityScapes Labels
    ├── Part_2         
    │   ├── train.py         
    │   ├── eval.py     
    │   ├── dataset.py           
    │   ├── network.py  
    │   ├── denseCRF.py  
    │   ├── model-FCN-city.cpt60    # Checkpoint file
    └── ...

To train the model
```
python3 train.py
```

To test the model
```
python3 eval.py
```



