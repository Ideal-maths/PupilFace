# PupilFace


Dataset

The dataset can be downloaded from the following link.   Extract code: 2yvx
https://pan.baidu.com/share/init?surl=xneLH0YtiDfO2oncesRY6w 

The step of Training

Use your own training weights
1.This paper uses the WiderFace dataset for training.
2.The WiderFace dataset can be downloaded from the above Baidu web drive.
3.Overwrite the ‘data’ folder in the root directory.
4.It is necessary to modify the code under train.py file. During training, it is necessary to pay attention to the correspondence between backbone and weight files.
5.The trained weight files are available in ‘logs’ folder.


Evalution steps

1.In retinaface.py file, modify “model_path” and “backbone” to match the trained files.
2.Download the data set uploaded on baidu net disk, including the verification set, decompressed in the root directory.
3.Run evaluation.py to begin the evaluation.

Reference
https://github.com/biubug6/Pytorch_Retinaface
https://github.com/bubbliiiing/retinaface-pytorch
