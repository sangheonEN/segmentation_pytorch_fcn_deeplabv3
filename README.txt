- Requirments
    → python 3.7
    → pytorch 1.7.1
    → cudatoolkit 11.0
    → pip install opencv-contrib-python
    → pip install -U albumentations
    → pip install tqdm
    → pip install pytz

- Training
    → python train.py --gpu_id=0 --mode trainval --backbone resnet --backbone_layer 50 --model fcn

- Inference
    → python inference.py --gpu_id=0 --mode inference --backbone resnet --backbone_layer 50 --model fcn --resume ckpt file path

The model applied pre-trained resnet 50 and 101 to the fcn and deeplabv3 segmentation models.
