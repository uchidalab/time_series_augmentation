# Augmentation Methods

All of the augmentation methods expect a numpy array ```x``` of size ```(batch, time_steps, channel)``` where ```batch``` is the size of the dataset or batch, ```time_steps``` is the number of time steps, and ```channel``` is the number of dimensions. Even if 1D time series are used, ```channel``` should still be 1. 


```from utils.augmentation import XXXX```

## Jittering
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

## Scaling
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

## Flipping/Rotation
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

## 2D Rotation
A special case of 2D rotation where the pattern is spacially rotated around the center.

```
rotation2d(x, sigma=0.2)
```

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

**sigma** : *float*

&nbsp;&nbsp;&nbsp;&nbsp; Standard deviation of the rotation amount.

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.

## Permutation

Random permutation of segments. A random number of segments is used, up to ```max_segments```.

```
permutation(x, max_segments=5, seg_mode="equal")
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

## Magnitude Warping

The magnitude of each time series is multiplied by a curve created by cubicspline with a set number of knots at random magnitudes.

```
magnitude_warp(x, sigma=0.2, knot=4)
```

Based on:
T. T. Um et al, "Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks," in ACM ICMI, pp. 216-220, 2017.

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

**sigma** : *float*

&nbsp;&nbsp;&nbsp;&nbsp; Standard deviation of the random magnitudes.

**knot** : *int*

&nbsp;&nbsp;&nbsp;&nbsp; Number of hills/valleys.

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.

## Time Warping

Random smooth time warping. 

```
time_warp(x, sigma=0.2, knot=4)
```

Based on:
T. T. Um et al, "Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks," in ACM ICMI, pp. 216-220, 2017.

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

**sigma** : *float*

&nbsp;&nbsp;&nbsp;&nbsp; Standard deviation of the random magnitudes of the warping path.

**knot** : *int*

&nbsp;&nbsp;&nbsp;&nbsp; Number of hills/valleys on the warping path.

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.

## Window Slicing

Cropping the time series by the ```reduce_ratio```. 

```
window_slice(x, reduce_ratio=0.9)
```

Based on:
A. Le Guennec, S. Malinowski, R. Tavenard, "Data Augmentation for Time Series Classification using
Convolutional Neural Networks," in ECML/PKDD Workshop on Advanced Analytics and Learning on Temporal Data, 2016.

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

**sigma** : *float*

&nbsp;&nbsp;&nbsp;&nbsp; Standard deviation of the random magnitudes of the warping path.

**knot** : *int*

&nbsp;&nbsp;&nbsp;&nbsp; Number of hills/valleys on the warping path.

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.

## Window Warping

Randomly warps a window by ```scales```.

```
window_warp(x, window_ratio=0.1, scales=[0.5, 2.])
```

Based on:
A. Le Guennec, S. Malinowski, R. Tavenard, "Data Augmentation for Time Series Classification using
Convolutional Neural Networks," in ECML/PKDD Workshop on Advanced Analytics and Learning on Temporal Data, 2016.

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

**window_ratio** : *float*

&nbsp;&nbsp;&nbsp;&nbsp; Ratio of the window to the full time series. 

**scales** : *list of floats*

&nbsp;&nbsp;&nbsp;&nbsp; A list ratios to warp the window by.

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.

## SuboPtimAl Warped time-series geNEratoR (SPAWNER)

Uses SPAWNER by K. Kamycki et al.

```
spawner(x, labels, sigma=0.05, verbose=0)
```

Based on:
K. Kamycki, T. Kapuscinski, M. Oszust, "Data Augmentation with Suboptimal Warping for Time-Series Classification," Sensors, vol. 20, no. 1, 2020.

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

**labels** : *2D or 3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Either list of integers or one hot of the labels.

**sigma** : *float*

&nbsp;&nbsp;&nbsp;&nbsp; Standard deviation of the jittering.

**verbose** : *int*

&nbsp;&nbsp;&nbsp;&nbsp; ```1``` prints out a DTW matrix. ```0``` shows nothing. 

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.

## Weighted Dynamic Time Warping Barycenter Averaging (wDBA)

Uses the Average Selected with Distance (ASD) version of DBA from G. Forestier et al.

```
wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True)
```

Based on:
G. Forestier, F. Petitjean, H. A. Dau, G. I. Webb, E. Keogh, "Generating Synthetic Time Series to Augment Sparse Datasets," in IEEE ICDM, 2017.

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

**labels** : *2D or 3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Either list of integers or one hot of the labels.

**batch_size** : *int*

&nbsp;&nbsp;&nbsp;&nbsp; How many patterns to average.

**slope_constraint** : *str*

&nbsp;&nbsp;&nbsp;&nbsp; Slope constraint for DTW. ```"symmetric"``` or ```"asymmetric"```. 

**use_window** : *bool*

&nbsp;&nbsp;&nbsp;&nbsp; Use a 10% boundary constraint window for DTW. ```True``` or ```False```. 

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.

## Random Guided Warping (RGW)

```
random_guided_warp(x, labels, slope_constraint="symmetric", use_window=True, dtw_type="normal")
```

Based on:
B. K. Iwana, S. Uchida, "Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher," arXiv, 2020.

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

**labels** : *2D or 3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Either list of integers or one hot of the labels.

**slope_constraint** : *str*

&nbsp;&nbsp;&nbsp;&nbsp; Slope constraint for DTW. ```"symmetric"``` or ```"asymmetric"```. 

**use_window** : *bool*

&nbsp;&nbsp;&nbsp;&nbsp; Use a 10% boundary constraint window for DTW. ```True``` or ```False```. 

**dtw_type** : *str*

&nbsp;&nbsp;&nbsp;&nbsp; Use DTW (```normal```) or shapeDTW (```shape```). 

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.

## Discriminative Guided Warping (DGW)

```
discriminative_guided_warp(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, dtw_type="normal", use_variable_slice=True)
```

Based on:
B. K. Iwana, S. Uchida, "Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher," arXiv, 2020.

##### Arguments
**x** : *3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of time series in format ```(batch, time_steps, channel)```.

**labels** : *2D or 3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Either list of integers or one hot of the labels.

**batch_size** : *int*

&nbsp;&nbsp;&nbsp;&nbsp; How many patterns to search. ```np.ceil(batch_size/2.)``` are used as positive samples and ```np.floor(batch_size/2.)``` are used for negative samples.

**slope_constraint** : *str*

&nbsp;&nbsp;&nbsp;&nbsp; Slope constraint for DTW. ```"symmetric"``` or ```"asymmetric"```. 

**use_window** : *bool*

&nbsp;&nbsp;&nbsp;&nbsp; Use a 10% boundary constraint window for DTW. ```True``` or ```False```. 

**dtw_type** : *str*

&nbsp;&nbsp;&nbsp;&nbsp; Use DTW (```normal```) or shapeDTW (```shape```). 

**use_variable_slice** : *bool*

&nbsp;&nbsp;&nbsp;&nbsp; Slice by the inverse of how much the pattern is warped. 

##### Returns
*3D numpy array*

&nbsp;&nbsp;&nbsp;&nbsp; Numpy array of generated data of equal size of the input ```x```.