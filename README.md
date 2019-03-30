# SFT-ReID

This is my implementation of the model proposed in paper [SFT-ReID](https://arxiv.org/abs/1811.11405).

My working environment is python3.5.2, and my pytorch version is 1.0.0. If things are not going well on your system, please check you environment.



### Get Market1501 dataset
Execute the script in the command line:
```
    $ sh get_market1501.sh
```
This will download the Market1501 dataset and extract to `dataset` directory.


### Train and Evaluate
* To train the model, just run the training script:  
```
    $ python train.py
```
This will train the model and save the parameters to the directory of ```res/```.

* To embed the gallery and query set with the trained model and compute the accuracy, directly run:
```
    $ python evaluate.py
```
This will embed the gallery and query set, and then compute cmc and mAP.  

Currently, I achieved accuracy of `93.20` rank-1, and `82.98` mAP without post-processing. If post-processing is added, the rank-1 and mAP can be `93.26` and `87.28`.

