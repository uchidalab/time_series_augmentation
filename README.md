# Time Series Augmentation

This is a Keras implementation of the paper, Brian Kenji Iwana and Seiichi Uchida, *Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher*, arXiv preprint arXiv:2004.08780, 2020. [LINK](https://arxiv.org/abs/2004.08780)

## News

- 2020/04/16: Repository Created.

## Requires

This code was developed in Python 3.5.2. and requires Tensorflow 1.10.0 and Keras 2.2.4

### Normal Install

```
pip install keras==2.2.4 numpy==1.14.5 matplotlib==2.2.2 scikit-image==0.15.0 tqdm
```

### Docker

```
cd docker
sudo docker build -t tsa .
docker run --runtime nvidia -rm -it -p 127.0.0.1:8888:8888 -v `pwd`:/work -w /work tsa jupyter notebook --allow-root
```

Newer docker installs might use ```--gpus all``` instead of ```--runtime nvidia```  

## Usage

### Description of Time Series Augmentation Methods

[Augmentation description](./docs/AugmentationMethods.md)

### Jupyter Example

[Jupyter Notebook](./example.ipynb)

### Keras Example

Example: 
To **train** a 1D **VGG** on the **FiftyWords** dataset from the **UCR Time Series Archive 2018** with **4x** the training dataset in **Jittering**, use:

```
python3 main.py --gpus=0 --dataset=FiftyWords --preset_files --ucr2018 --normalize_input --train --save --jitter --augmentation_ratio=4 --model=vgg
```

## Citation

B. K. Iwana and S. Uchida, "Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher," arXiV, 2020.

```
@article{iwana2020time,
  title={Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher},
  author={Iwana, Brian Kenji and Uchida, Seiichi},
  journal={arXiv preprint 2004.08780},
  year={2020}
}
```
