# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions related to preprocessing inputs."""
import tensorflow as tf
import cv2
import numpy as np


def flip_dim(image, label, prob=0.5, dim=1):
  """Randomly flips a dimension of the given tensor.

  The decision to randomly flip the `Tensors` is made together. In other words,
  all or none of the images pass in are flipped.

  Note that tf.random_flip_left_right and tf.random_flip_up_down isn't used so
  that we can control for the probability as well as ensure the same decision
  is applied across the images.

  Args:
    tensor_list: A list of `Tensors` with the same number of dimensions.
    prob: The probability of a left-right flip.
    dim: The dimension to flip, 0, 1, ..

  Returns:
    outputs: A list of the possibly flipped `Tensors` as well as an indicator
    `Tensor` at the end whose value is `True` if the inputs were flipped and
    `False` otherwise.

  Raises:
    ValueError: If dim is negative or greater than the dimension of a `Tensor`.
  """
  random_value = np.random.uniform()

  def flip(image):

      if dim < 0 or dim >= len(image.shape):
          raise ValueError('dim must represent a valid dimension.')
      flipped = cv2.flip(image, flipCode=dim)
      return flipped

  is_flipped = random_value <= prob
  out_img = flip(image) if is_flipped else image
  out_label = flip(label) if is_flipped else label

  return out_img, out_label


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_value):
  """Pads the given image with the given pad_value.

  Works like tf.image.pad_to_bounding_box, except it can pad the image
  with any given arbitrary pad value and also handle images whose sizes are not
  known during graph construction.

  Args:
    image: 3-D tensor with shape [height, width, channels]
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.
    pad_value: Value to pad the image tensor with.

  Returns:
    3-D tensor of shape [target_height, target_width, channels].

  Raises:
    ValueError: If the shape of image is incompatible with the offset_* or
    target_* arguments.
  """
  if len(image.shape) != 3 and len(image.shape) != 2:
      raise ValueError('Wrong image rank')
  image = image.astype(np.float) -  pad_value
  height, width = image.shape[0:2]
  if target_width < width:
      raise ValueError('target_width must be >= width')
  if target_height < height:
      raise ValueError('target_height must be >= height')
  after_padding_width = target_width - offset_width - width
  after_padding_height = target_height - offset_height - height
  if after_padding_height < 0 or after_padding_width < 0:
      raise ValueError('target size not possible with the given target offsets')

  padded = cv2.copyMakeBorder(image, offset_height, after_padding_height, offset_width, after_padding_width,cv2.BORDER_CONSTANT)

  return padded + pad_value


def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    The cropped (and resized) image.

  Raises:
    ValueError: if `image` doesn't have rank of 3.
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = image.shape
  if len(image.shape) != 3 and len(image.shape) != 2:
      raise ValueError('Wrong image rank')
  if original_shape[0] < crop_height or original_shape[1] < crop_width:
      raise ValueError('Crop size greater than the image size.')

  crop_image = image[offset_height:offset_height+crop_height, offset_width:offset_width+crop_width]

  return crop_image


def random_crop(image, label, crop_height, crop_width):
  """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """

  # Compute the rank assertions.
  if len(image.shape) != 3 and len(image.shape) != 2:
      raise ValueError('Wrong image rank')

  image_height, image_width = image.shape[0:2]
  if image_height < crop_height or image_width < crop_width:
      raise ValueError('Crop size greater than the image size.')


  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  max_offset_height = image_height - crop_height
  max_offset_width = image_width - crop_width
  offset_height = np.random.random_integers(0, max_offset_height)
  offset_width = np.random.random_integers(0, max_offset_width)

  return _crop(image, offset_height, offset_width,crop_height, crop_width), _crop(label, offset_height, offset_width,crop_height, crop_width)


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
  """Gets a random scale value.

  Args:
    min_scale_factor: Minimum scale value.
    max_scale_factor: Maximum scale value.
    step_size: The step size from minimum to maximum value.

  Returns:
    A random scale value selected between minimum and maximum value.

  Raises:
    ValueError: min_scale_factor has unexpected value.
  """
  if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
    raise ValueError('Unexpected value of min_scale_factor.')

  if min_scale_factor == max_scale_factor:
    return float(min_scale_factor)

  # When step_size = 0, we sample the value uniformly from [min, max).
  if step_size == 0:
    return np.random.uniform(min_scale_factor, max_scale_factor, [1])

  # When step_size != 0, we randomly select one discrete value from [min, max].
  num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
  scale_factors = np.linspace(min_scale_factor, max_scale_factor, num_steps)
  np.random.shuffle(scale_factors)
  return scale_factors[0]


def randomly_scale_image_and_label(image, label=None, scale=1.0):
  """Randomly scales image and label.

  Args:
    image: Image with shape [height, width, 3].
    label: Label with shape [height, width, 1].
    scale: The value to scale image and label.

  Returns:
    Scaled image and label.
  """
  # No random scaling if scale == 1.
  if scale == 1.0:
    return image, label
  image_shape = image.shape
  new_dim = tuple([int(e * scale) for e in image_shape[0:2]])

  # Need squeeze and expand_dims because image interpolation takes
  # 4D tensors as input.
  image = cv2.resize(image, new_dim)
  if label is not None:
    label = cv2.resize(label, new_dim, interpolation=cv2.INTER_NEAREST)

  return image, label


'''def resolve_shape(tensor, rank=None, scope=None):
  """Fully resolves the shape of a Tensor.

  Use as much as possible the shape components already known during graph
  creation and resolve the remaining ones during runtime.

  Args:
    tensor: Input tensor whose shape we query.
    rank: The rank of the tensor, provided that we know it.
    scope: Optional name scope.

  Returns:
    shape: The full shape of the tensor.
  """
  with tf.name_scope(scope, 'resolve_shape', [tensor]):
    if rank is not None:
      shape = tensor.get_shape().with_rank(rank).as_list()
    else:
      shape = tensor.get_shape().as_list()

    if None in shape:
      shape_dynamic = tf.shape(tensor)
      for i in range(len(shape)):
        if shape[i] is None:
          shape[i] = shape_dynamic[i]

    return shape'''


def resize_to_range(image,
                    label=None,
                    min_size=None,
                    max_size=None,
                    factor=None,
                    label_layout_is_chw=False):
  """Resizes image or label so their sizes are within the provided range.

  The output size can be described by two cases:
  1. If the image can be rescaled so its minimum size is equal to min_size
     without the other side exceeding max_size, then do so.
  2. Otherwise, resize so the largest side is equal to max_size.

  An integer in `range(factor)` is added to the computed sides so that the
  final dimensions are multiples of `factor` plus one.

  Args:
    image: A 3D tensor of shape [height, width, channels].
    label: (optional) A 3D tensor of shape [height, width, channels] (default)
      or [channels, height, width] when label_layout_is_chw = True.
    min_size: (scalar) desired size of the smaller image side.
    max_size: (scalar) maximum allowed size of the larger image side. Note
      that the output dimension is no larger than max_size and may be slightly
      smaller than min_size when factor is not None.
    factor: Make output size multiple of factor plus one.
    align_corners: If True, exactly align all 4 corners of input and output.
    label_layout_is_chw: If true, the label has shape [channel, height, width].
      We support this case because for some instance segmentation dataset, the
      instance segmentation is saved as [num_instances, height, width].
    scope: Optional name scope.
    method: Image resize method. Defaults to tf.image.ResizeMethod.BILINEAR.

  Returns:
    A 3-D tensor of shape [new_height, new_width, channels], where the image
    has been resized (with the specified method) so that
    min(new_height, new_width) == ceil(min_size) or
    max(new_height, new_width) == ceil(max_size).

  Raises:
    ValueError: If the image is not a 3D tensor.
  """
  new_list = []
  min_size = float(min_size)
  if max_size is not None:
    max_size = float(max_size)
      # Modify the max_size to be a multiple of factor plus 1 and make sure the
      # max dimension after resizing is no larger than max_size.
    if factor is not None:
      max_size = (max_size + (factor - (max_size - 1) % factor) % factor
                  - factor)

  orig_height, orig_width, _ = image.shape
  orig_height = float(orig_height)
  orig_width = float(orig_width)
  orig_min_size = np.minimum(orig_height, orig_width)

    # Calculate the larger of the possible sizes
  large_scale_factor = min_size / orig_min_size
  large_height = int(np.ceil(orig_height * large_scale_factor))
  large_width = int(np.ceil(orig_width * large_scale_factor))
  large_size = np.stack([large_height, large_width])

  new_size = large_size
  if max_size is not None:
      # Calculate the smaller of the possible sizes, use that if the larger
      # is too big.
    orig_max_size = np.maximum(orig_height, orig_width)
    small_scale_factor = max_size / orig_max_size
    small_height = int(np.ceil(orig_height * small_scale_factor))
    small_width = int(np.ceil(orig_width * small_scale_factor))
    small_size = np.stack([small_height, small_width])
    new_size = small_size if float(tf.reduce_max(large_size)) > max_size else large_size
    # Ensure that both output sides are multiples of factor plus one.
  if factor is not None:
      new_size += (factor - (new_size - 1) % factor) % factor
  new_list.append(cv2.resize(image, new_size))
  if label is not None:
      if label_layout_is_chw:
        # Input label has shape [channel, height, width].
        resized_label = np.expand_dims(label, 3)
        resized_label = cv2.resize(resized_label, new_size ,interpolation=cv2.INTER_NEAREST)
        resized_label = np.squeeze(resized_label, 3)
      else:
        # Input label has shape [height, width, channel].
        resized_label = cv2.resize(resized_label, new_size ,interpolation=cv2.INTER_NEAREST)
      new_list.append(resized_label)
  else:
      new_list.append(None)
  return new_list
