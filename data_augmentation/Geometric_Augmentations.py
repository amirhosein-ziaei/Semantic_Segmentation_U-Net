import tensorflow as tf
from tensorflow import Tensor
from typing import Tuple

def horizontal_flip(image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    flip_prob = tf.random.uniform(())
    
    if flip_prob > 0.5:
        flipped_image = tf.image.flip_left_right(image)
        flipped_mask = tf.image.flip_left_right(mask)

    return flipped_image, flipped_mask