import tensorflow as tf
from tensorflow import Tensor
from typing import Tuple


def horizontal_flip(image: Tensor, mask: Tensor, probability: float = 0.5) -> Tuple[Tensor, Tensor]:
    """
    Apply horizontal flip data augmentation to an input image and its corresponding mask.

    Args:
        image (Tensor): The input image tensor.
        mask (Tensor): The input mask tensor.
        probability (float, optional): The probability of applying the horizontal flip. Defaults to 0.5.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the horizontally flipped image and mask tensors.
    """
    
    flip_prob = tf.random.uniform(())
    
    if flip_prob >= probability:
        flipped_image = tf.image.flip_left_right(image)
        flipped_mask = tf.image.flip_left_right(mask)
        return flipped_image, flipped_mask

    return image, mask


def vertical_flip(image: Tensor, mask: Tensor, probability: float = 0.5) -> Tuple[Tensor, Tensor]:
    """
    Perform vertical flip (up-down flip) on the input image and mask tensors using TensorFlow.

    Args:
        image (Tensor): The input image tensor.
        mask (Tensor): The input mask tensor.
        probability (float, optional): The probability of applying the vertical flip. Defaults to 0.5, which means a 50% chance of flipping.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the vertically flipped image and mask tensors.
    """
    
    flip_prob = tf.random.uniform(())

    if flip_prob >= probability:
        flipped_image = tf.image.flip_up_down(image)
        flipped_mask = tf.image.flip_up_down(mask)

        return flipped_image, flipped_mask

    return image, mask

def rotaion(image: Tensor, mask: Tensor, max_angle: int = 30, probability: float = 0.5) -> Tuple[Tensor, Tensor]:
    """
    Rotate the input image and mask by a random angle within the specified range using TensorFlow.

    Args:
        image (Tensor): The input image tensor.
        mask (Tensor): The input mask tensor.
        max_angle (int, optional): The maximum angle of rotation in degrees. Defaults to 30 degrees.
        probability (float, optional): The probability of applying the vertical flip. Defaults to 0.5.

    Returns:
        Tuple[Tensor, tf.Tensor]: A tuple containing the rotated image and rotated mask tensors.
    """
    flip_prob = tf.random.uniform(())
    
    if flip_prob >= probability:
        angle = tf.random.uniform(shape=[], minval=-max_angle, maxval=max_angle, dtype=tf.float32) # output is scalar tensor

        angle_rad = tf.math.divide(angle * tf.constant(3.14159265358979323846), tf.constant(180.0))

        rotated_image = tf.image.rot90(image, k=tf.cast(angle_rad, dtype=tf.int32) // 90)
        rotated_mask = tf.image.rot90(mask, k=tf.cast(angle_rad, dtype=tf.int32) // 90)
        
        return rotated_image, rotated_mask

    return image, mask
    