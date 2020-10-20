Fit to Measure: a size-aware object recognition system
======================================================

[![image](https://img.shields.io/pypi/v/object_reasoner.svg)](https://pypi.python.org/pypi/object_reasoner)
[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Welcome! This repository provides you with a Command Line Interface to
incorporate reasoning about the typical size of objects in your object
recognition project.

:robot: :heart: Our application focus is on **service robotics**,
[here](http://robots.kmi.open.ac.uk/) you can find more info about how
this work fits into our broader ideas.

For more information, you can check out our paper:

> **Fit to Measure: combining the knowledge of object sizes with Machine
> Learning for robust object recognition**
> ([arxiv]())
>
>[Agnese Chiatti](https://achiatti.github.io/), [Enrico Motta](http://people.kmi.open.ac.uk/motta/),
> [Enrico Daga](http://www.enridaga.net/about/), [Gianluca Bardaro](http://kmi.open.ac.uk/people/member/gianluca-bardaro)

Tech frameworks
---------------

The Machine Learning models used here were developed in
[PyTorch](https://pytorch.org/). We also provide some handy utils to
process [ROSbag data](http://wiki.ros.org/rosbag/Code%20API#Python_API),
depth images (via [OpenCV](https://opencv.org/)) and PointClouds
(through the [Open3D](http://www.open3d.org) library).

Features
--------

![image](assets/framework.svg?raw=true)

-   **ML-based object recognition.** supported models (see also the `MLonly/models.py` script):
    -   Nearest Neighbour similarity matching of ResNet50 embeddings
            pre-trained on ImageNet
    -   the N-net multi-branch Network of [Zeng et al., 2018](https://arxiv.org/abs/1710.01330)
    -   the K-net multi-branch Network of [Zeng et al., 2018](https://arxiv.org/abs/1710.01330)
    -   K-net with weight imprinting in the SoftMax layer ([Chiatti
            et al.,2020](https://www.mdpi.com/2079-9292/9/3/380))

-   **Knowledge-based reasoner**. Reasoning is used after applying ML to generate a first set of predictions.
    Specifically, we estimate the real size of an object based on depth data and then infer a set of candidate
    classes which are plausible from the standpoint of size. This validation step is used to correct the ML predictions.
    Ultimately, the validated prediction which maximises the ML similarity score is picked to classify the object.

Supported datasets
------------------

-   the KMi dataset (see instructions below)
-   the 2017 Amazon Robotic Challenge (ARC) image matching set - please
    refer to these
    [instructions](https://github.com/andyzeng/arc-robot-vision/tree/master/image-matching/)

The KMi Dataset
---------------
* **Train-validation RGB set**: includes 60 object classes commonly found at the Knowledge Media Institute (KMi).
It is conceived for a few-shot metric learning scenario: only 4 images per class are devoted to training and 1 image for validation.
For each class, a support set of 5 reference images is also provided. Triplets are formed directly
at training time (you can refer to the code at `./object_reasoner/MLonly/data_loaders.py` and to our paper for more details).
This set is already available under `./object_reasoner/data/KMi-set-2020`

* **RGB-D test dataset**: includes **1414 object regions** (polygonal masks
or rectangular bounding boxes, depending on the object). For each RGB region,
also the matching Depth image region is provided. Objects in this test set belong to 47 of
the 60 object classes. Annotations follow the same text formatting as the [ARC2017 image
matching set](https://github.com/andyzeng/arc-robot-vision/tree/master/image-matching/). Instructions to download this larger dataset are in the "Getting started" section.


* **KMi size catalogue**: we also provide ground truth size annotations for all 60 classes,
in csv and JSON format (under `./object_reasoner/data`). The reasoning modules expect the JSON catalogue as input, so
we also provide a script to convert raw csv data to JSON, in case you needed to repeat the
steps for your own data/set of classes. The size representation is multi-dimensional and categorises
objects qualitatively, based on their surface area, depth, and Aspect Ratio (AR), as exemplified in the below picture:

![image](assets/size_representation.svg?raw=true)

Installation
------------
**Tested on Ubuntu 18.04**

* Pypi dependencies can be installed through pip.
  If re-training on a GPU-enabled machine, change the last line to install torch & torchvision for GPU

  ```
    sudo apt install python3-pip
    pip3 install --upgrade pip
    pip3 install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
    cd your_path_to/object_reasoner
    pip3 install -r requirements_dev.txt
    pip3 install .
   ```
* It is recommended to build Open3D from source: step-by-step instructions can be found on the [official docs](http://www.open3d.org/docs/release/compilation.html). Note: no C++ installation required, you can build the package
for python3 by making cmake point to Python3 (e.g.: `cmake -DPYTHON_EXECUTABLE=/usr/bin/python3 ..`)
and then using the `make pip-package` before installing the python lib manually with pip3:
    ```
    cd open3d/build/lib/python_package
    pip3 install .
    ```
* **[OPTIONAL]** The `./object_reasoner/preprocessing/bag_processing.py` is ROS dependent and was tested on ROS melodic. The barebone ROS installation is sufficient to run this code. The reference instructions to install ROS melodic on Ubuntu can be found [here](http://wiki.ros.org/melodic/Installation/Ubuntu)

Getting started
---------------
* After completing all the installation steps, it is time to download our starter kit!
  **Note: this step requires about 21 GB of free disk space**.
  The kit can be downloaded [here](http://www.mediafire.com/file/df2upaslfrpbd0d/starter_kit.zip/file) and includes the KMi test RGB-D set as well as the pre-trained
  models to rerun our pipeline directly for inference.

* After downloading and unzipping the starter kit:
    * Move or copy the baselineNN, imprk-net, k-net and n-net folders under `object_reasoner/data`
    * Move or copy the remaining files and folders (i.e., KMi test set) under `object_reasoner/data/KMi-set-2020`

Command examples
----------------
To reproduce inference results on KMi set (as reported in our paper).
```
cd your_path_to/object_reasoner/object_reasoner
```
* Realistic scenario (correcting selected predictions, based on ML confidence):
    ```
    python3 cli.py ./data/KMi-set-2020 ./data
    ```
* Best-case scenario (correcting only those predictions which need correction, based on ground truth):
    ```
    python3 cli.py ./data/KMi-set-2020 ./data --scenario best
    ```
* Worst-case scenario (correcting only those predictions which need correction, based on ground truth):
    ```
    python3 cli.py ./data/KMi-set-2020 ./data --scenario worst
    ```
You can also run the following for other combinations (e.g., starting from a different ML baseline, or on the ARC set):
```
python3 cli.py --help
```
For usage details on how to re-train or produce new ML-based predictions on a different dataset,
you can run the following commands:
```
cd your_path_to/object_reasoner/object_reasoner
python3 MLonly/main.py --help
```
Credits
-------

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
