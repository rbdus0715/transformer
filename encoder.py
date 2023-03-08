import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        '''
        단어 임베딩 벡터에 더해주어 위치 정보를 부여함
        
        1. get_angles
        하나의 문장은 2차원으로 나타낼 수 있다. (문장의 길이, 임베딩 차원)
        get_angles의 목적은 위치 정보를 부여하기 위한 (문장의 길이, 임베딩 차원) size의 위치 테이블을 만들어 제공하는 것
        
        반환하는 값 : position * angles
        position : (문장의 길이, 1)
        angles : (1, 임베딩 벡터 차원 크기)
        return size : (max_len, embedding_size)

        2. positional_encoding
        get_angles에서 넘겨받은 테이블을 사용하여 짝수 임베딩 차원에는 sin, 홀수에는 cos을 적용하여 반환한다.
        마지막에 앞에 차원 하나를 추가하여 반환
        ex) (50, 128) -> (1, 50, 128)

        + 사용 예시
        pos_encoding = PositionalEncoding(입력 단어 갯수, 임베딩 차원)
        '''
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2*(i//2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position = tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model = d_model
        )

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        # 한 문장 길이 만큼만
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]





def scaled_dot_product_attention(query, key, value, mask):
    '''
    인코더에의 입력으로 (seq_len, embedding_size) 크기의 문장이 들어오면
    그것을 num_heads 수 만큼으로 나누어 멀티헤드어텐션을 수행한다. (논문에서는 num_heads를 8로 설정)
    
    예를 들어 (seq_len, 128)로 임베딩 된 문장으로 보면
    num_heads를 8로 하면 하나 당 사이즈는 (seq_len, 128/8) = (seq_len, 16)

    이 함수에서는 그 나눠진 상태에서의 쿼리, 키, 값을 인풋으로 사용한다.
    이후 이 함수의 아웃풋들을 모두 연결(concatenate)한다.
    즉, 다시 (seq_len, 128) 사이즈로 만든다.
    '''
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast((key.shape)[-1],dtype=tf.float32)
    temp = matmul_qk / tf.math.sqrt(depth)

    if mask is not None:
        temp += (mask * -1e9)

    attention_weights = tf.nn.softmax(temp, axis=-1)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights





class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, 
                d_model, 
                num_heads,
                name="multi_head_attention"
                ):
        '''
        멀티헤드 어텐션
        한 문장의 입력을 batch_size까지 표현하면 (1, max_len, embedding_size) 이다
        논문에서는 max_len => position, embedding_size => d_model 로 표현하고 있지만 이해를 위해서 위와 같이 쓴다.
        query, key, value는 각각 다른 matrix와 다른 값의 dense 레이어를 거친다. 즉, 같은 연산을 독립적으로 계산한다.
        
        query 기준으로, 
        (1) query : (batch, max_len, embedding_size) 
        (2) query vector : query * dense => (batch, max_len, embedding_size)
        (3) split_heads 과정 : 
            (batch, max_len, embedding_size) => (batch, max_len, num_heads, depth)
            (embedding_size = num_heads * depth)
            => (batch, num_heads, max_len, depth)
            즉, num_heads 개의 (max_len, depth)로 쪼개는 작업
        (4) scaled_dot_product_attention 이후 다시 (batch, max_len, embedding_size)로 concatenate 시킴
        '''
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth)
        )
        return tf.transpose(inputs, perm=[0,2,1,3])
    
    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query[0])

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)        

        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3])

        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model)
        )
        
        outputs = self.dense(concat_attention)

        return outputs





def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x,0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis]





def encoder_layer(dff, 
                  d_model, 
                  num_heads, 
                  dropout, 
                  name="encoder_layer"):
    
    # 입력
    inputs = tf.keras.Onput(shape=(None, d_model), name="inputs")
    padding_mask =  tf.keras.Input(shape=(1,1,None), name="padding_mask")
    
    # 서브층 (1) : 셀프 어텐션
    attention = MultiHeadAttention(d_model, num_heads, name="attention")
    ({
        'query':inputs, 'key':inputs, 'value':inputs,
        'mask':padding_mask
    })

    # 드롭아웃, residual
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6
    )(inputs + attention)

    # 서브층 (2) : 포지션 와이드 피드 포워드 신경망
    outputs = tf.keras.layers.Dense(units=dff, activate='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # 드롭아웃, residual
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6
    )(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)





def encoder(vocab_size, 
            num_layers, 
            dff,
            d_model,
            num_heads,
            dropout,
            name="encoder"
            ):
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    padding_mask = tf.keras.Input(shape=(1,1,None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, 
                                d_model=d_model, 
                                num_heads=num_heads, 
                                dropout=dropout,
                                name="encoder_layer_{}".format(i)
                                )([outputs, padding_mask])
        
    return tf.keras.Model(
        inputs = [inputs, padding_mask], outputs = outputs, name = name
    )



#####======== 디코더 ========#####
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
    return tf.maximum(look_ahead_mask, padding_mask)





def decoder_layer(vocab_size, 
                  num_layers, 
                  dff,
                  d_model,
                  num_heads,
                  dropout,
                  name='decoder'
                  ):
    
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

    look_ahead_mask = tf.keras.Input(
        shape=(1,None,None), name='look_ahead_mask'
    )
    padding_mask = tf.keras.Input(shape=(1,1,None), name='padding_mask')

    attention1 = MultiHeadAttention(d_model, num_heads, name="attention1")(
        inputs={
        'query':inputs, 'key':inputs, 'value':inputs,
        'mask':look_ahead_mask
        }
    )

    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6
    )(attention1 + inputs)

    attention2 = MultiHeadAttention(d_model, num_heads, name="attention2")(
        inputs={
        'query':attention1, 'key':enc_outputs, 'value':enc_outputs,
        'mask':padding_mask
        }
    )

    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6
    )(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6
    )(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name
    )





def decoder(vocab_size,
            num_layers,
            dff,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

    look_ahead_mask = tf.keras.Input(
        shape=(1,None,None), name='look_ahead_mask'
    )
    padding_mask = tf.kears.Input(shape=(1,1,None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(dff=dff,
                                d_model=d_model,
                                num_heads=num_heads,
                                dropout=dropout,
                                name='decoder_layer_{}'.format(i),
                                )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])
        
    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name
    )

