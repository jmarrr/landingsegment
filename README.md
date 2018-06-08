# Landing Segmentation

# Getting Started

* [inspect_data.ipynb](inspect_data.ipynb) This notebook visualizes the different pre-processing steps to prepare the training data.

* [landing.ipynb](landing.ipynb) This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline.

* [process_video.py](process_video.py) This file is used to process video and apply segmentation 

* To test on a video
   ```
    python process_video.py --video={PATH TO VIDEO}
    ``` 

## Requirements
* Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.


## Installation
1. Install dependencies
   ```
   pip install -r requirements.txt
   ```
2. Clone this repository
3. Run setup from the repository root directory
    ```
    python setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).

5. Most recent pre-trained weights specified for landing download it [here](https://www.dropbox.com/s/6x7qhlrs60nmmu0/logs.zip?dl=0) 

6. Dataset used get it [here](https://www.dropbox.com/s/6ond7mmfhwvdb02/dataset.zip?dl=0)


# Training on Your Own Dataset

Start by reading this [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). It covers the process starting from annotating images to training to using the results in a sample application.

In summary, to train the model on your own dataset you'll need to extend two classes:

```Config```
This class contains the default configuration. Subclass it and modify the attributes you need to change.

```Dataset```
This class provides a consistent way to work with any dataset. 
It allows you to use new datasets for training without having to change 
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not 
all available in one dataset. 







