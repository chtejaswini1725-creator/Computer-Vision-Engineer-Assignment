#### Description

This data contains images with twenty different types of objects.

```python
{'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}
```

Each image can contain one or more ground truth objects.

Each object is represented by –

- a bounding box in absolute boundary coordinates

- a label (one of the object types mentioned above)

-  a perceived detection difficulty (either `0`, meaning _not difficult_, or `1`, meaning _difficult_)

#### Download

Specifically, you will need to download the following VOC datasets –

- [2007 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (460MB)

- [2012 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (2GB)

- [2007 _test_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) (451MB)

Consistent with the paper, the two _trainval_ datasets are to be used for training, while the VOC 2007 _test_ will serve as our test data.  

Make sure you extract both the VOC 2007 _trainval_ and 2007 _test_ data to the same location, i.e. merge them.

### Inputs to model

We will need three inputs.

#### Images

Since we're using the SSD300 variant, the images would need to be sized at `300, 300` pixels and in the RGB format.

Remember, we're using a VGG-16 base pretrained on ImageNet that is already available in PyTorch's `torchvision` module. [This page](https://pytorch.org/docs/master/torchvision/models.html) details the preprocessing or transformation we would need to perform in order to use this model – pixel values must be in the range [0,1] and we must then normalize the image by the mean and standard deviation of the ImageNet images' RGB channels.

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Also, PyTorch follows the NCHW convention, which means the channels dimension (C) must precede the size dimensions.

Therefore, **images fed to the model must be a `Float` tensor of dimensions `N, 3, 300, 300`**, and must be normalized by the aforesaid mean and standard deviation. `N` is the batch size.

#### Objects' Bounding Boxes

We would need to supply, for each image, the bounding boxes of the ground truth objects present in it in fractional boundary coordinates `(x_min, y_min, x_max, y_max)`.

Since the number of objects in any given image can vary, we can't use a fixed size tensor for storing the bounding boxes for the entire batch of `N` images.

Therefore, **ground truth bounding boxes fed to the model must be a list of length `N`, where each element of the list is a `Float` tensor of dimensions `N_o, 4`**, where `N_o` is the number of objects present in that particular image.

#### Objects' Labels

We would need to supply, for each image, the labels of the ground truth objects present in it.

Each label would need to be encoded as an integer from `1` to `20` representing the twenty different object types. In addition, we will add a _background_ class with index `0`, which indicates the absence of an object in a bounding box. (But naturally, this label will not actually be used for any of the ground truth objects in the dataset.)

Again, since the number of objects in any given image can vary, we can't use a fixed size tensor for storing the labels for the entire batch of `N` images.

Therefore, **ground truth labels fed to the model must be a list of length `N`, where each element of the list is a `Long` tensor of dimensions `N_o`**, where `N_o` is the number of objects present in that particular image.

### Data pipeline

As you know, our data is divided into _training_ and _test_ splits.

#### Parse raw data

See `create_data_lists()` in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py).

This parses the data downloaded and saves the following files –

- A **JSON file for each split with a list of the absolute filepaths of `I` images**, where `I` is the total number of images in the split.

- A **JSON file for each split with a list of `I` dictionaries containing ground truth objects, i.e. bounding boxes in absolute boundary coordinates, their encoded labels, and perceived detection difficulties**. The `i`th dictionary in this list will contain the objects present in the `i`th image in the previous JSON file.

- A **JSON file which contains the `label_map`**, the label-to-index dictionary with which the labels are encoded in the previous JSON file. This dictionary is also available in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py) and directly importable.

#### PyTorch Dataset

See `PascalVOCDataset` in [`datasets.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py).

This is a subclass of PyTorch [`Dataset`](https://pytorch.org/docs/master/data.html#torch.utils.data.Dataset), used to **define our training and test datasets.** It needs a `__len__` method defined, which returns the size of the dataset, and a `__getitem__` method which returns the `i`th image, bounding boxes of the objects in this image, and labels for the objects in this image, using the JSON files we saved earlier.

You will notice that it also returns the perceived detection difficulties of each of these objects, but these are not actually used in training the model. They are required only in the [Evaluation](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#evaluation) stage for computing the Mean Average Precision (mAP) metric. We also have the option of filtering out _difficult_ objects entirely from our data to speed up training at the cost of some accuracy.

Additionally, inside this class, **each image and the objects in them are subject to a slew of transformations** as described in the paper and outlined below.

#### Data Transforms

See `transform()` in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py).

This function applies the following transformations to the images and the objects in them –

- Randomly **adjust brightness, contrast, saturation, and hue**, each with a 50% chance and in random order.

- With a 50% chance, **perform a _zoom out_ operation** on the image. This helps with learning to detect small objects. The zoomed out image must be between `1` and `4` times as large as the original. The surrounding space could be filled with the mean of the ImageNet data.

- Randomly crop image, i.e. **perform a _zoom in_ operation.** This helps with learning to detect large or partial objects. Some objects may even be cut out entirely. Crop dimensions are to be between `0.3` and `1` times the original dimensions. The aspect ratio is to be between `0.5` and `2`. Each crop is made such that there is at least one bounding box remaining that has a Jaccard overlap of either `0`, `0.1`, `0.3`, `0.5`, `0.7`, or `0.9`, randomly chosen, with the cropped image. In addition, any bounding boxes remaining whose centers are no longer in the image as a result of the crop are discarded. There is also a chance that the image is not cropped at all.

- With a 50% chance, **horizontally flip** the image.

- **Resize** the image to `300, 300` pixels. This is a requirement of the SSD300.

- Convert all boxes from **absolute to fractional boundary coordinates.** At all stages in our model, all boundary and center-size coordinates will be in their fractional forms.

- **Normalize** the image with the mean and standard deviation of the ImageNet data that was used to pretrain our VGG base.

As mentioned in the paper, these transformations play a crucial role in obtaining the stated results.

#### PyTorch DataLoader

The `Dataset` described above, `PascalVOCDataset`, will be used by a PyTorch [`DataLoader`](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) in `train.py` to **create and feed batches of data to the model** for training or evaluation.

Since the number of objects vary across different images, their bounding boxes, labels, and difficulties cannot simply be stacked together in the batch. There would be no way of knowing which objects belong to which image.

Instead, we need to **pass a collating function to the `collate_fn` argument**, which instructs the `DataLoader` about how it should combine these varying size tensors. The simplest option would be to use Python lists.

### Base Convolutions

See `VGGBase` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, we **create and apply base convolutions.**

The layers are initialized with parameters from a pretrained VGG-16 with the `load_pretrained_layers()` method.

We're especially interested in the lower-level feature maps that result from `conv4_3` and `conv7`, which we return for use in subsequent stages.

### Auxiliary Convolutions

See `AuxiliaryConvolutions` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, we **create and apply auxiliary convolutions.**

Use a [uniform Xavier initialization](https://pytorch.org/docs/stable/nn.html#torch.nn.init.xavier_uniform_) for the parameters of these layers.

We're especially interested in the higher-level feature maps that result from `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`, which we return for use in subsequent stages.

### Prediction Convolutions

See `PredictionConvolutions` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, we **create and apply localization and class prediction convolutions** to the feature maps from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`.

These layers are initialized in a manner similar to the auxiliary convolutions.

We also **reshape the resulting prediction maps and stack them** as discussed. Note that reshaping in PyTorch is only possible if the original tensor is stored in a [contiguous](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.contiguous) chunk of memory.

As expected, the stacked localization and class predictions will be of dimensions `8732, 4` and `8732, 21` respectively.

### Putting it all together

See `SSD300` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, the **base, auxiliary, and prediction convolutions are combined** to form the SSD.

There is a small detail here – the lowest level features, i.e. those from `conv4_3`, are expected to be on a significantly different numerical scale compared to its higher-level counterparts. Therefore, the authors recommend L2-normalizing and then rescaling _each_ of its channels by a learnable value.

### Priors

See `create_prior_boxes()` under `SSD300` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

This function **creates the priors in center-size coordinates** as defined for the feature maps from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`, _in that order_. Furthermore, for each feature map, we create the priors at each tile by traversing it row-wise.

This ordering of the 8732 priors thus obtained is very important because it needs to match the order of the stacked predictions.

### Multibox Loss

See `MultiBoxLoss` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Two empty tensors are created to store localization and class prediction targets, i.e. _ground truths_, for the 8732 predicted boxes in each image.

We **find the ground truth object with the maximum Jaccard overlap for each prior**, which is stored in `object_for_each_prior`.

We want to avoid the rare situation where not all of the ground truth objects have been matched. Therefore, we also **find the prior with the maximum overlap for each ground truth object**, stored in `prior_for_each_object`. We explicitly add these matches to `object_for_each_prior` and artificially set their overlaps to a value above the threshold so they are not eliminated.

Based on the matches in `object_for_each prior`, we set the corresponding labels, i.e. **targets for class prediction**, to each of the 8732 priors. For those priors that don't overlap significantly with their matched objects, the label is set to _background_.

Also, we encode the coordinates of the 8732 matched objects in `object_for_each prior` in offset form `(g_c_x, g_c_y, g_w, g_h)` with respect to these priors, to form the **targets for localization**. Not all of these 8732 localization targets are meaningful. As we discussed earlier, only the predictions arising from the non-background priors will be regressed to their targets.

The **localization loss** is the [Smooth L1 loss](https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss) over the positive matches.

Perform Hard Negative Mining – rank class predictions matched to _background_, i.e. negative matches, by their individual [Cross Entropy losses](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss). The **confidence loss** is the Cross Entropy loss over the positive matches and the hardest negative matches. Nevertheless, it is averaged only by the number of positive matches.

The **Multibox Loss is the aggregate of these two losses**, combined in the ratio `α`. In our case, they are simply being added because `α = 1`.

# Training

Before you begin, make sure to save the required data files for training and evaluation. To do this, run the contents of [`create_data_lists.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/create_data_lists.py) after pointing it to the `VOC2007` and `VOC2012` folders in your [downloaded data](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#download).

See [`train.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/train.py).

The parameters for the model (and training it) are at the beginning of the file, so you can easily check or modify them should you need to.

To **train your model from scratch**, run this file –

`python train.py`

To **resume training at a checkpoint**, point to the corresponding file with the `checkpoint` parameter at the beginning of the code.

### Remarks

In the paper, they recommend using **Stochastic Gradient Descent** in batches of `32` images, with an initial learning rate of `1e−3`, momentum of `0.9`, and `5e-4` weight decay.

I ended up using a batch size of `8` images for increased stability. If you find that your gradients are exploding, you could reduce the batch size, like I did, or clip gradients.

The authors also doubled the learning rate for bias parameters. As you can see in the code, this is easy do in PyTorch, by passing [separate groups of parameters](https://pytorch.org/docs/stable/optim.html#per-parameter-options) to the `params` argument of its [SGD optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD).

The paper recommends training for 80000 iterations at the initial learning rate. Then, it is decayed by 90% (i.e. to a tenth) for an additional 20000 iterations, _twice_. With the paper's batch size of `32`, this means that the learning rate is decayed by 90% once after the 154th epoch and once more after the 193th epoch, and training is stopped after 232 epochs. I followed this schedule.

On a TitanX (Pascal), each epoch of training required about 6 minutes.

I should note here that two unintended differences from the paper were brought to my attention by readers of this tutorial:

- My priors that overshoot the edges of the image are not being clipped, as pointed out in issue [#94](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/issues/94) by _@AakiraOtok_. This does not appear to have a negative effect on performance, however, as discussed in that issue and also verified in issue [#95](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/issues/95) by the same reader. It is even possible that there is a slight improvement in performance, but this may be too small to be conclusive.

- I mistakenly used L1 loss instead of *smooth* L1 loss as the localization loss, as pointed out in issue [#60](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/issues/60) by _jonathan016_. This also appears to have no negative effect on performance as pointed out in that issue, but _smooth_ L1 loss may offer better training stability with larger batch sizes as mentioned in [this comment](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/issues/94#issuecomment-1590217018). 

### Model checkpoint

You can download this pretrained model [here](https://drive.google.com/open?id=1bvJfF6r_zYl2xZEpYXxgb7jLQHFZ01Qe).

Note that this checkpoint should be [loaded directly with PyTorch](https://pytorch.org/docs/stable/torch.html?#torch.load) for evaluation or inference – see below.

# Evaluation

See [`eval.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/eval.py).

The data-loading and checkpoint parameters for evaluating the model are at the beginning of the file, so you can easily check or modify them should you wish to.

To begin evaluation, simply run the `evaluate()` function with the data-loader and model checkpoint. **Raw predictions for each image in the test set are obtained and parsed** with the checkpoint's `detect_objects()` method, which implements [this process](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#processing-predictions). Evaluation has to be done at a `min_score` of `0.01`, an NMS `max_overlap` of `0.45`, and `top_k` of `200` to allow fair comparision of results with the paper and other implementations.

**Parsed predictions are evaluated against the ground truth objects.** The evaluation metric is the _Mean Average Precision (mAP)_. If you're not familiar with this metric, [here's a great explanation](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173).

We will use `calculate_mAP()` in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py) for this purpose. As is the norm, we will ignore _difficult_ detections in the mAP calculation. But nevertheless, it is important to include them from the evaluation dataset because if the model does detect an object that is considered to be _difficult_, it must not be counted as a false positive.

The model scores **77.2 mAP**, same as the result reported in the paper.

Class-wise average precisions (not scaled to 100) are listed below.

| Class | Average Precision |
| :-----: | :------: |
| _bicycle_ | 0.8351995348930359 |
| _car_ | 0.8655831217765808 |
| _dog_ | 0.856262743473053 |
| _person_ | 0.7884440422058105 |
| _train_ | 0.8655905723571777 |

You can see that some objects, like bottles and potted plants, are considerably harder to detect than others.

# Inference

See [`detect.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/detect.py).

Point to the model you want to use for inference with the `checkpoint` parameter at the beginning of the code.

Then, you can use the `detect()` function to identify and visualize objects in an RGB image.

```python
img_path = '/path/to/ima.ge'
original_image = PIL.Image.open(img_path, mode='r')
original_image = original_image.convert('RGB')

detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
```

This function first **preprocesses the image by resizing and normalizing its RGB channels** as required by the model. It then **obtains raw predictions from the model, which are parsed** by the `detect_objects()` method in the model. The parsed results are converted from fractional to absolute boundary coordinates, their labels are decoded with the `label_map`, and they are **visualized on the image**.

There are no one-size-fits-all values for `min_score`, `max_overlap`, and `top_k`. You may need to experiment a little to find what works best for your target data.

### Some more examples

---

<p align="center">
<img src="./img/000029.jpg">
</p>

---

<p align="center">
<img src="./img/000045.jpg">
</p>

---

<p align="center">
<img src="./img/000062.jpg">
</p>

---

<p align="center">
<img src="./img/000075.jpg">
</p>

---

<p align="center">
<img src="./img/000085.jpg">
</p>

---

<p align="center">
<img src="./img/000092.jpg">
</p>

---

<p align="center">
<img src="./img/000100.jpg">
</p>

---

<p align="center">
<img src="./img/000124.jpg">
</p>

---

<p align="center">
<img src="./img/000127.jpg">
</p>

---

<p align="center">
<img src="./img/000128.jpg">
</p>

---

<p align="center">
<img src="./img/000145.jpg">
</p>

---
