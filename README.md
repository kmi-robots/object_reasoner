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

-   **ML-based object recognition.** supported models (see also models.py script):
    -   Nearest Neighbour similarity matching of ResNet50 embeddings
            pre-trained on ImageNet
    -   the N-net multi-branch Network of [Zeng et al., 2018](https://arxiv.org/abs/1710.01330)
    -   the K-net multi-branch Network of [Zeng et al., 2018](https://arxiv.org/abs/1710.01330)
    -   K-net with weight imprinting in the SoftMax layer ([Chiatti
            et al.,2020](https://www.mdpi.com/2079-9292/9/3/380))

-   **Knowledge-based reasoner**

Supported datasets
------------------

-   the KMi dataset (see instructions below)
-   the 2017 Amazon Robotic Challenge (ARC) image matching set - please
    refer to these
    [instructions](https://github.com/andyzeng/arc-robot-vision/tree/master/image-matching/)

The KMi Dataset
---------------

RGB-D and knowledge catalogue Links for download

Proposed size representation

![image](assets/size_representation.svg?raw=true)

Installation
------------
**Tested on Ubuntu 18.04**


* Pypi dependencies can be installed through pip.
  If re-training on a GPU-enabled machine, change the last line to install torch & torchvision for GPU

  ```
    sudo apt install python3-pip
    cd your_path_to/object_reasoner
    pip3 install .
    pip3 install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
   ```

* The ./object_reasoner/preprocessing/bag_processing.py is ROS dependent and to be run requires 
  to install ROS. We tested it on ROS melodic. The reference instructions to install ROS on Ubuntu can be found [here](http://wiki.ros.org/melodic/Installation/Ubuntu)


Command examples
----------------

How to reproduce the results in the paper
Test commands

Commands to re-train or produce new predictions with ML.

Credits
-------

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.