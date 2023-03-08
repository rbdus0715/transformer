import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


max_len  = 50
embedding_size = 128

"위치 정보"
position = tf.range(max_len, dtype=tf.float32)
# >> [0, 1, 2, 3, 4, ..., 49]
position = position[:, tf.newaxis]
# >> [[0.], [1.], ..., [49.]] 
# dimension : (50, 1)

"임베딩 차원의 인덱스"
i = tf.range(embedding_size, dtype=tf.float32)[tf.newaxis, :]
# >> [[0., 1., ..., 127.]]
# demension : (1, 128)

"""
(1) x = 2 * (i//2)
i : 0 1 2 3 4 5 ... 126 127
x : 0 0 2 2 4 4 ... 126 126

(2) x / embedding_size
= 0 ~ 1 사이의 값으로 normalize

(3) 10000^(0~1의 값들)
= 최소 1 ~ 최대 10000의 값들

(4) 역수
= 1부터 계속해서 작아지는 수들
"""

angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(embedding_size, tf.float32))
angle_rads = position * angles
"""
position.shape, angles.shape
>> (50, 1) (1, 128)
# position
[
[0],
[1],
...
[49]
]
# angles
[[1, 1, ..., 0에 가까워짐]]
position * angles
>> (50, 128)

# dimension(위치 정보, 임베딩 차원)
"""

