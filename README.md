# 3D-Object-Detection-and-Recognition

![Python](https://img.shields.io/badge/python-v3.7-blue)
<a href="https://docs.python.org/3/library/os.html"><img src="https://img.shields.io/badge/os-orange?style=flat&logo=os"></a>
<a href="https://trimsh.org/trimesh.html"><img src="https://img.shields.io/badge/trimesh-darkblue?style=flat&logo=os"></a>
<a href="https://numpy.org/"><img src="https://img.shields.io/badge/numpy-yellogreen?style=flat&logo=numpy&labelColor=yellogreen"></a>
<a href="https://docs.python.org/3/library/glob.html"><img src="https://img.shields.io/badge/glob-skyblue?style=flat&logo=os"></a>
<a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/matplotlib-lightgreen?style=flat&logo=matplotlib"></a>
<a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/tensorflow-white?style=flat&logo=tensorflow&labelColor=white"></a>
<a href="https://keras.io/"><img src="https://img.shields.io/badge/keras-darkred?style=flat&logo=keras&labelColor=darkred"></a>
<a href="https://tqdm.github.io/"><img src="https://img.shields.io/badge/tqdm-purple?style=flat&logo=tqdm&labelColor=purple"></a>
<a href="https://pytorch.org/get-started/locally/"><img src="https://img.shields.io/badge/torch-darkgreen?style=flat&logo=pytorch&labelColor=darkgreen"></a>
<a href="https://kaolin.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/kaolin-violet?style=flat&logo=kaolin"></a>


> A general 3D object detection approach in Python.

---

### Table of Contents
Following sections will help you to navigate to any section.

- [Introduction](#introduction)
- [Description](#description)
- [Motivation](#motivation)
- [How To Use](#how-to-use)
- [References](#references)
- [License](#license)
- [Author Info](#author-info)

---

## Introduction

Object detection and recogntion is a demanding application in the area of machine learning and computer vision. Object detection using 3D techniques have beneficial impacts to mitigate real-world applications such as robotics, self-driving cars, security and many more.  There are several 3D object detection algorithms available to implement accurate object detection system to accomplish real-time delicate tasks. Nowadays, object detection using LiDAR point clouds or 3D representation of data has become neccesary to avoid imprecision in complex or unstructued scenes. This project is implemented to achive high accurate results in detecting objects using Graph Convolutional Neural Networks and 3D representation of data.

## Description

This project is using a 3D object detection approach without the prerequisite for particular operating system or Graphics Processing Unit (GPU) in hand. As we know that 3D data offers well founded depth information to locate or detect any object accurately. These days, Point cloud representation is more favourable, as it can preserve the original depth information in 3D space without lossing any precious information. To add to it, point clouds representation is also convenient to be transformed to other file formats. A project is using a open-source Kaolin PyTorch framework to detect the objects accurately using 3D models. This framework is very approachable in loading and processing 3D datasets. 

In the project, a model is trained using ModelNet10 datasets including total 10 3D object categories to train high quality deep network. A model is traned using a graph convolutional network. A complete project is implemented by following total five stages including data loading, data pre-processing, training, evaluating and testing. A project has used the follwoing technologies to implement an excellent object detection application in order to evaluate the real-world significant problems.

## Motivation

These days, object detection and recognization is a challenging and difficult task to achive ideal results in the field of machine learrning and computer vision. So, 2D object detection using traditional techniques of object detection is not efficient to solve real-time problems. So, in order to solve this kinds of numerous issues 3D object detection and recognition has become a primary step in the field of machine learning to innovate different products that endorse human beings to ease their works in smart ways. Behind all the discussions, a real motivation is for me to explore some advanced 3D object detection techniques and to implement this project to achive the highest accuracy using data with 3D representation.

---

## Dependencies

In order to use this project, you need to have installed the following dependencies :

* Trimesh
* NumPy
* Matplotlib
* Tensorflow
* Keras
* Tqdm
* PyTorch
* Kaolin

You can install all the mentioned dependencies by running the below commands

**Trimesh**
```bash
pip install trimesh
```

**NumPy**
```bash
pip install numpy
```

**Matplotlib**
```bash
pip install matplotlib
```

**Tensorflow**
```bash
pip install tensorflow==2.4.0
```

or **Tensorflow GPU** if you have NVIDIA GPU with installed CUDA and cuDNN.
```bash
pip install tensorflow-gpu==2.4.0
```

**Keras**
```bash
pip install keras
```

**Tqdm**
```bash
pip install tqdm
```

**PyTorch**
```bash
pip install torch
```

**Kaolin**
```bash
pip install git+https://github.com/NVIDIAGameWorks/kaolin.git
```

#### Installation

#### API Reference

```html
    <p>dummy code</p>
```
[Back To The Top](#read-me-template)

---

## References
[Back To The Top](#read-me-template)

---

## License

MIT License

Copyright (c) [2017] [James Q Quick]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[Back To The Top](#read-me-template)

---

## Author Info

- Twitter - [@jamesqquick](https://twitter.com/jamesqquick)
- Website - [James Q Quick](https://jamesqquick.com)

[Back To The Top](#read-me-template)


[python-image]: https://img.shields.io/badge/Python-3.6-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.5-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
