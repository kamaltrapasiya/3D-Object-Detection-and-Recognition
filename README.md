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


> A general 3D object detection approach in Python

---

### Table of Contents
Following sections will help you to navigate to any section.

- [Introduction](#introduction)
- [Description](#description)
- [Motivation](#motivation)
- [Dependencies](#dependencies)
- [Platform](#platform)
- [Dataset](#dataset)
- [Kaolin: A PyTorch Framework](#Kaolin-A-PyTorch-Framework)
- [Graph Convolutional Networks](#Graph-Convolutional-Networks)
- [Results](#results)
- [Support](#support)
- [License](#license)
- [References](#references)

---

## Introduction

Object detection and recognition is a demanding application in the area of machine learning and computer vision. Object detection using 3D techniques have beneficial impacts to mitigate real-world applications such as robotics, self-driving cars, security and many more.  There are several 3D object detection algorithms available to implement accurate object detection systems to accomplish real-time delicate tasks. Nowadays, object detection using LiDAR point clouds or 3D representation of data has become necessary to avoid imprecision in complex or unstructured scenes. This project is implemented to achieve high accurate results in detecting objects using Graph Convolutional Neural Networks and 3D representation of data.

![flow](https://user-images.githubusercontent.com/37270872/129836235-811974e7-1167-459f-942b-e5676fd76fc3.png)

As we can see, the above figure depicts the working flow of general object detection and recognition processes. According to this figure, image features are generated while training the training images, and all images are labelled to classify further. After that in the testing phase, from the digital images or videos, image features are extracted and classified to predict the object.

## Description

This project is using a 3D object detection approach without the prerequisite for a particular operating system or Graphics Processing Unit (GPU) in hand. As we know that 3D data offers well founded depth information to locate or detect any object accurately. These days, Point cloud representation is more favourable, as it can preserve the original depth information in 3D space without losing any precious information. To add to it, point cloud representation is also convenient to be transformed to other file formats. A project is using an open-source Kaolin PyTorch framework to detect the objects accurately using 3D models. This framework is very approachable in loading and processing 3D datasets. 

In the project, a model is trained using ModelNet10 datasets including total 10 3D object categories to train high quality deep networks. A model is trained using a graph convolutional network. A complete project is implemented by following a total five stages including data loading, data pre-processing, training, evaluating and testing. A project has used the following technologies to implement an excellent object detection application in order to evaluate the real-world significant problems.


## Motivation

These days, object detection and recognition is a challenging and difficult task to achieve ideal results in the field of machine learning and computer vision. So, 2D object detection using traditional techniques of object detection is not efficient to solve real-time problems. So, in order to solve these kinds of issues, 3D object detection and recognition has become a primary step in the field of machine learning to innovate different products that endorse human beings to ease their work in smart ways. Behind all the discussions, a real motivation is for me to explore some advanced 3D object detection techniques and to implement this project to achieve the highest accuracy using data with 3D representation.

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
pip install tensorflow
```

or **Tensorflow GPU** if you have NVIDIA GPU with installed CUDA and cuDNN.
```bash
pip install tensorflow-gpu
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
---

## Platform

This project is implemented using Google Colab. If you want to use this project with Google Colab, you can directly open it from below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kamaltrapasiya/3D-Object-Detection-and-Recognition/blob/add-license-1/3D%20Object%20Detection.ipynb)

---

## Dataset

In this project, a proposed method of object detection is implemented using a ModelNet dataset. This  exceptional  dataset  offers two kinds of datasets including ModelNet10 and ModelNet40. A ModelNet10 dataset contains a total 10 categories of CAD models to train the deep neural network, and a ModelNet40 dataset consists of a total 40 categories of CAD models to train the deep neural network. This project is utilizing the ModelNet10 dataset to speed up the performance in the training  process to come up with the solution in a less amount of time. Moreover, training and testing data is already split in the ModelNet dataset, so we do not need to split the data for training and testing by utilizing split methods. A ModelNet10 dataset contains 10 categories, namely toilet, sofa, table, dresser, monitor, night stand, desk, bed, chair and bath tub. A ModelNet dataset consists of object files with object file format (OFF) that provides the information about geometry architecture of the object.

---

## Kaolin: A PyTorch Framework

This project is implemented using a Kaolin framework which is an open-source PyTorch library developed by NVIDIA following the 3D deep learning approach. Kaolin provides very convenient ways to load and preprocess the dataset without any issue. A Kaolin framework offers numerous built-in algorithms and techniques to ease our work in the application of object detection and recognition in the field of machine learning and computer vision. A primary goal of the Kaolin is to provide several easy-to-use tools in constructing 3D deep learning architectures.

---

## Graph Convolutional Networks

In this project, a model is trained using a graph convolutional network. Graph Convolutional Networks (GCNs) are the effective variant of the convolutional neural networks on graphs. Graph Convolutional Networks (GCNs) learn a new feature representation by considering neighboring nodes for the multifarious features as similar as convolutional neural networks. Following figure represents the architecture of the Graph Convolutional Networks (GCNs).

![graph](https://user-images.githubusercontent.com/37270872/129837495-9965ddff-a739-40fb-8722-1d0e80153606.png)

In graph convolutional networks, input neurons with a set of weights are multiplied, and it is also called kernels or filters. Kernels or filters proceed as a sliding window within the image and authorize convolutional neural networks to learn the different features from the neighboring nodes. A primary benefit of applying graph convolutional networks is that class labels are organized perfectly in the graph structure to understand the neural network easily.

---

## Results

I have experimented with various 3D methods and techniques, but a proposed technique has achieved exceptional results in detecting 3D objects. A training process of the model has consumed around 20-25 minutes, as the model will work for only 10 objects with providing the highest accuracy. Surprisingly, the trained model has provided around 93.34% accuracy in detecting 3D objects in the testing phase. Moreover, I have scheduled and tried experiments several times, and I came to know that I was able to detect 30 objects out of 32 objects perfectly. So, I have achieved an expected result in the object detection process by employing the graph convolutional neural network.

![72wZ](https://user-images.githubusercontent.com/37270872/130000081-f34349fc-3a10-4fc3-878e-c6b385ca725c.gif)


---

## Support

![twitter](https://img.shields.io/twitter/follow/kamaltrapasiya1?style=social)

Contact: [Email me](mailto:kamaltrapasiya97@gmail.com?subject=[GitHub]%20Source%20Han%20Sans)

---

## License

MIT License

Copyright (c) [2021] [Kamalkumar Trapasiya]

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

---

### References
 <div id="ref"></div>

 1. Stanford artificial Intelligence Laboratory. (n.d.). https://ai.stanford.edu/~syyeung/cvweb/tutorial4.html.
 2. Jatavallabhula, Krishna Murthy, et al. "Kaolin: A pytorch library for accelerating 3d deep learning research." arXiv preprint arXiv:1911.05063 (2019).
 3. NVIDIAGameWorks. (n.d.). NVIDIAGameWorks/kaolin: A Pytorch library for Accelerating 3D deep learning research. GitHub. https://github.com/NVIDIAGameWorks/kaolin. 
 4. The Trustees of Princeton University. (n.d.). Princeton MODELNET. Princeton University. https://modelnet.cs.princeton.edu/#.
 5. NVIDIAGameWorks. (n.d.). NVIDIAGameWorks/kaolin: A Pytorch library for Accelerating 3D deep learning research. GitHub. https://github.com/NVIDIAGameWorks/kaolin.
 6. Jatavallabhula, Krishna Murthy, et al. "Kaolin: A pytorch library for accelerating 3d deep learning research." arXiv preprint arXiv:1911.05063 (2019).
 7. Zhang, S., Tong, H., Xu, J. et al. Graph convolutional networks: a comprehensive review. Comput Soc Netw 6, 11 (2019). https://doi.org/10.1186/s40649-019-0069-y
 8. Mayachita, I. (2020, August 18). Understanding graph convolutional networks for node classification. Medium. https://towardsdatascience.com/understanding-graph-convolutional-networks-for-node-classification-a2bfdb7aba7b.


[python-image]: https://img.shields.io/badge/Python-3.6-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.5-2BAF2B.svg
[pytorch-url]: https://pytorch.org/

