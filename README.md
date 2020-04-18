# Time Series Augmentation

This is a Keras implementation of the paper, Brian Kenji Iwana and Seiichi Uchida, *Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher*.

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

### Augmentation Methods

All of the augmentation methods expect a numpy array ```x``` of size ```(batch, time_steps, channel)``` where ```batch``` is the size of the dataset or batch, ```time_steps``` is the number of time steps, and ```channel``` is the number of dimensions. Even if 1D time series are used, ```channel``` should still be 1. 

#### Jittering
Adding jittering, or noise, to the time series.

```
jitter(x, sigma=0.03)
```

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

**sigma** : *float*

&nbsp;&nbsp;&nbsp;&nbsp; Standard deviation of the distribution to be added.

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.

#### Scaling
Scaling each time series by a constant amount.

```
scaling(x, sigma=0.1)
```

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

**sigma** : *float*

&nbsp;&nbsp;&nbsp;&nbsp; Standard deviation of the scaling constant.

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.

#### Flipping/Rotation
For 1D time series, randomly flipping. For multivariate time series, flipping as well as axis shuffling.

```
rotation(x)
```

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.

#### 2D Rotation
A special case of 2D rotation where the pattern is spacially rotated around the center.

```
rotation2d(x, sigma=0.2):
```

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

**sigma** : *float*

&nbsp;&nbsp;&nbsp;&nbsp; Standard deviation of the rotation amount.

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.

#### Permutation

Random permutation of segments. A random number of segments is used, up to ```max_segments```.

```
permutation(x, max_segments=5, seg_mode="equal"):
```

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

**max_segments** : *int*

&nbsp;&nbsp;&nbsp;&nbsp; The maximum number of segments to use. The minimum number is 1.

**seg_mode** : *str*

&nbsp;&nbsp;&nbsp;&nbsp; ```equal``` uses equal sized segments and ```random``` uses randomly sized segments.

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.

#### Magnitude Warping

#### Time Warping

#### Window Slicing

#### Window Warping

#### SuboPtimAl Warped time-series geNEratoR (SPAWNER)

#### Weighted Dynamic Time Warping Barycenter Averaging (wDBA)

#### Random Guided Warping (RGW)

#### Discriminative Guided Warping (DGW)

### Jupyter Example


### Keras Example


## Citation

B. K. Iwana and S. Uchida, "Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher," arXiV, 2020.

```
@article{iwana2020time,
  title={Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher},
  author={Iwana, Brian Kenji and Uchida, Seiichi},
  journal={arXiV},
  year={2020}
}
```
