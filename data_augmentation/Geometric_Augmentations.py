import tensorflow as tf
from tensorflow import Tensor
from typing import Tuple

def horizontal_flip(image: Tensor, mask: Tensor, probability: int = 0.5) -> Tuple[Tensor, Tensor]:
    """
    Apply horizontal flip data augmentation to a car image and its corresponding mask.

    Args:
        image (Tensor): The car image tensor.
        mask (Tensor): The mask image tensor.
        probability (int, optional): The probability of applying the horizontal flip. Defaults to 0.5.


    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the horizontally flipped car image and mask image.
    """
    flip_prob = tf.random.uniform(())
    
    if flip_prob > probability:
        flipped_image = tf.image.flip_left_right(image)
        flipped_mask = tf.image.flip_left_right(mask)

    return flipped_image, flipped_mask


def vertical_flip(image: Tensor, mask: Tensor, probability: int = 0.5) -> Tuple[Tensor, Tensor]:
    flip_prob = tf.random.uniform(())

    if flip_prob > probability:
        flipped_image = tf.image.flip_up_down(image)
        flipped_mask = tf.image.flip_up_down(mask)

    return flipped_image, flipped_mask