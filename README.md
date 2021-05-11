# Time Series Augmentation

This is a collection of time series data augmentation methods and an example use using Keras.

## News

- 2020/04/16: Repository Created.
- 2020/06/22: Accepted to ICPR 2020 - B. K. Iwana and S. Uchida, *Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher*, ICPR 2020 [LINK](https://arxiv.org/abs/2004.08780)
- 2020/07/31: Survey Paper Posted on arXiv - B. K. Iwana and S. Uchida *An Empirical Survey of Data Augmentation for Time Series Classification
  with Neural Networks*, arXiv [LINK](https://arxiv.org/abs/2007.15951)
- 2021/05/11: Tensorflow v1 branched. The master will now support Tensorflow v2.

## Requires

This code was developed in Python 3.6.9. and requires Tensorflow 2.4.1 and Keras 2.2.4

### Normal Install

```
pip install tensorflow==2.4.1 keras==2.2.4 numpy==1.19.5 matplotlib==2.2.2 scikit-image==0.15.0 tqdm
```

### Docker

```
cd docker
sudo docker build -t tsa .
docker run --runtime nvidia -rm -it -p 127.0.0.1:8888:8888 -v `pwd`:/work -w /work tsa jupyter notebook --allow-root
```

Newer docker installs might use ```--gpus all``` instead of ```--runtime nvidia```  

### Dataset

`main.py` was designed to use the UCR Time Series Archive 2018 datasets. To install the datasets, download the .zip file from https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ and extract the contents into the `data` folder.

## Usage

### Description of Time Series Augmentation Methods

[Augmentation description](./docs/AugmentationMethods.md)

### Jupyter Example

[Jupyter Notebook](./example.ipynb)

### Keras Example

Example: 
To **train** a 1D **VGG** on the **FiftyWords** dataset from the **UCR Time Series Archive 2018** with **4x** the training dataset in **Jittering**, use:

```
python3 main.py --gpus=0 --dataset=CBF --preset_files --ucr2018 --normalize_input --train --save --jitter --augmentation_ratio=4 --model=vgg
```

## Citation

B. K. Iwana and S. Uchida, "An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks," arXiv, 2020.

```
@article{iwana2020empirical,
  title={An Empirical Survey of Data Augmentation for Time Series Classification
  with Neural Networks},
  author={Iwana, Brian Kenji and Uchida, Seiichi},
  journal={arXiv preprint arXiv:2007.15951},
  year={2020}
}
```
