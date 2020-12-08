Code docs
======================================================

[![image](https://img.shields.io/pypi/v/object_reasoner.svg)](https://pypi.python.org/pypi/object_reasoner)
[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Supported datasets
------------------

-   the Lab RGBD dataset: You can find the relevant data and models to reproduce
    our results as part of a zipped folder [here](https://mega.nz/file/klcH3SiQ#mxzEyejzJDwl0WRoN-YOFrwHG8Qb-hCcqi04qqda7k4).
    **Note: the extracted files require about 2.5 GB of free disk space**.
-   the 2017 Amazon Robotic Challenge (ARC) image matching set: please
    refer to these [instructions](https://github.com/andyzeng/arc-robot-vision/tree/master/image-matching/).
    **Note: the zipped dataset occupies 4.6 GB**.

- The size knowledge properties generated for both datasets are saved in JSON format under
 ```./object_reasoner/data/```

Tech frameworks
---------------

The Machine Learning models used here were developed in
[PyTorch](https://pytorch.org/). We also provide some handy utils to
process [ROSbag data](http://wiki.ros.org/rosbag/Code%20API#Python_API),
depth images (via [OpenCV](https://opencv.org/)) and PointClouds
(through the [Open3D](http://www.open3d.org) library).


Installing Package Dependencies
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

Getting started
---------------
* After completing all the installation steps, download the relevant data [here](https://mega.nz/file/klcH3SiQ#mxzEyejzJDwl0WRoN-YOFrwHG8Qb-hCcqi04qqda7k4).

* After downloading and unzipping the starter kit:
    * Move ```arc_camera_intrinsics.txt``` under ```./arc-robot-vision/image-matching/``` and rename it as ```camera-intrinsics.txt```
    * Move the ```snapshot-no-class``` and ```snapshot-with-class``` folders under ```./arc-robot-vision/image-matching/```. These folders contain
        the results of the k-net and n-net models as produced in the original repo by Zeng et al.
    * Move the k-net and n-net folders under `object_reasoner/object_reasoner/data`
    * Move the remaining files and folders under `object_reasoner/object_reasoner/data/Lab-set`

*   Clone [repository](https://github.com/andyzeng/arc-robot-vision) of baseline methods in cited paper by Zeng et al.

*   Download the ARC data as described in the arc-robot-vision repository (Note: as specified in Zeng et al's instructions,
    ARC data will have to be placed under a ```./arc-robot-vision/image-matching/data``` folder)


Reproducing paper results
----------------
Commands to reproduce our results on the Lab test set:
```
cd your_path_to/object_reasoner/object_reasoner
python3 cli.py ./data/Lab-set ./data
```
Commands to reproduce our results on the ARC test set:
```
cd your_path_to/object_reasoner/object_reasoner
python3 cli.py your_path_to/arc-robot-vision/image-matching/data your_path_to/arc-robot-vision/image-matching --baseline two-stage --set arc --preds your_path_to/arc-robot-vision/image-matching/data/logged-predictions
```

For usage details on how to re-train or produce new ML-based predictions on a different dataset,
you can run the following commands:
```
cd your_path_to/object_reasoner/object_reasoner
python3 MLonly/main.py --help
```
