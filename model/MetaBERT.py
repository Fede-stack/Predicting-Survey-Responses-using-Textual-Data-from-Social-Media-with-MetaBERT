import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, MultiHeadAttention, Dense, Concatenate, Softmax, Dot, Add, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, losses

class WeightedSum(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.w = None

    def build(self, input_shape):
        self.w = self.add_weight(
            name='weights',
            shape=(len(input_shape),),
            initializer='ones',
            trainable=True
        )
        super(WeightedSum, self).build(input_shape)

    def call(self, inputs):
        weighted_sum = tf.reduce_sum([self.w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        return weighted_sum

class ReduceSumLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ReduceSumLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)




class MetaGPTModel:
    def __init__(self, embedding_dim=768, max_posts=20, num_choices=4, learning_rate=1e-3):
        self.embedding_dim = embedding_dim
        self.max_posts = max_posts
        self.num_choices = num_choices
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        #input layers
        input_reddit = Input((self.max_posts, self.embedding_dim))
        input_description = Input((self.embedding_dim, ))

        input_desc = Reshape((1, self.embedding_dim))(input_description)

        #cross-attention layer
        cross_attention_layer = MultiHeadAttention(key_dim=self.embedding_dim, num_heads=2)
        cross_attention_, cross_weights = cross_attention_layer(input_desc, input_reddit, return_attention_scores=True)
        cross_attention_ = Reshape((self.embedding_dim, ))(cross_attention_)

        Att_Reddit = Attention()([Input_Reddit, Input_Reddit, Input_Reddit])
        Pooling_Posts = ReduceSumLayer(axis=1)(Att_Reddit)
      
        #cosine similarity between reddit posts contents and item's choices
        inputs_choices = []
        output_similarity = []
        for i in range(self.num_choices):
            inp_choice_i = Input((self.embedding_dim, ))
            inputs_choices.append(inp_choice_i)
            cosine_sim = Dot(axes=-1, normalize=True)([cross_attention_, inp_choice_i])
            output_similarity.append(cosine_sim)

        concatenated_similarities = Concatenate()(output_similarity)

        #first output
        softmax_similarities = Softmax(name='soft')(concatenated_similarities)

        #weighted sum of reddit post representations based on item's choice similarity
        weighted_sum = Add()([Multiply()([inp_choice, softmax_similarities[:, i:i+1]]) for i, inp_choice in enumerate(inputs_choices)])
      
        merge_layer = weighted_sum  

        # second output
        out_soft_1 = Dense(4, activation='softmax', name='out')(merge_layer)

        predictions_concatenate = Concatenate(axis=-1)([out_soft_1, softmax_similarities])

        #third output - meta predictions
        meta_predictions = Softmax(name='meta')(predictions_concatenate)

        #model identification
        meta_gpt = Model(inputs=[input_reddit, input_description] + inputs_choices, outputs=[out_soft_1, softmax_similarities, meta_predictions])

        meta_gpt.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                         loss={'out': losses.SparseCategoricalCrossentropy(), 'soft': losses.SparseCategoricalCrossentropy(), 'meta': losses.SparseCategoricalCrossentropy()},
                         loss_weights={'out': 0.33, 'soft': 0.33, 'meta': 0.33})
        
        return meta_gpt

    def get_model(self):
        return self.model

    def summary(self):
        self.model.summary()

#example:
# model_instance = MetaGPTModel()
# model_instance.summary()
# model = model_instance.get_model()
