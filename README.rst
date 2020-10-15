======================================================
Fit to Measure: a size-aware object recognition system
======================================================

.. image:: https://img.shields.io/pypi/v/object_reasoner.svg
        :target: https://pypi.python.org/pypi/object_reasoner
.. image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
    :target: https://www.python.org/
.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0


Welcome! This repository provides you with a Command Line Interface
to incorporate reasoning about the typical size of objects
in your object recognition project.

Our application focus is on **service robotics**, here_ you can find more info about how this work fits into our broader ideas.

For more information, you can check out our paper:

    **Fit to Measure: combining the knowledge of object sizes with Machine Learning for robust object recognition** (arxiv_)

    `Agnese Chiatti`_, `Enrico Motta`_, `Enrico Daga`_, `Gianluca Bardaro`_

.. _here: http://robots.kmi.open.ac.uk/
.. _arxiv:
.. _`Agnese Chiatti`: https://achiatti.github.io/
.. _`Enrico Motta`: http://people.kmi.open.ac.uk/motta/
.. _`Enrico Daga`: http://www.enridaga.net/about/
.. _`Gianluca Bardaro`: http://kmi.open.ac.uk/people/member/gianluca-bardaro

Tech frameworks
---------------
The Machine Learning models used here were developed in PyTorch_.
We also provide some handy utils to process `ROSbag data`_, depth images (via OpenCV_)
and PointClouds (through the Open3D_ library).

.. _PyTorch: https://pytorch.org/
.. _`ROSbag data`: http://wiki.ros.org/rosbag/Code%20API#Python_API
.. _OpenCV: https://opencv.org/
.. _Open3D: http://www.open3d.org

Features
--------
.. image:: https://raw.githubusercontent.com/kmi-robots/object-reasoner/master/object_reasoner/assets/framework.png

- **ML-based object recognition.** supported models (see also models.py script):
   - Nearest Neighbour similarity matching of ResNet50 embeddings pre-trained on ImageNet
   - the N-net multi-branch Network of Zeng et al., 2018
   - the K-net multi-branch Network of Zeng et al., 2018
   - K-net with weight imprinting in the SoftMax layer (Chiatti et al.,2020)

- **Knowledge-based reasoner**

Supported datasets
------------------

- the KMi set (see instructions below)
- the 2017 Amazon Robotic Challenge (ARC) image matching set - please refer to these instructions_

.. _instructions: https://github.com/andyzeng/arc-robot-vision/tree/master/image-matching/

The KMi Dataset
---------------------------
RGB-D and knowledge catalogue
Links for download

Proposed size representation

.. image:: https://raw.githubusercontent.com/kmi-robots/object-reasoner/master/object_reasoner/assets/size_representation.png

Installation
-------------
Dependencies and installation steps

Command examples
----------------
How to reproduce the results in the paper


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
