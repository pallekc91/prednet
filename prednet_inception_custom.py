from keras.layers import Recurrent
from keras.engine import InputSpec
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from keras import backend
from keras import activations


class PredNetInceptionCustom(Recurrent):

    def __init__(self, channels_a, channels_r, glob_filter_size, output_mode='error', **kwargs):
        super(PredNetInceptionCustom, self).__init__(**kwargs)
        self.conv_layers = {c: [] for c in ['i1', 'f1', 'c1', 'o1','i3', 'f3', 'c3', 'o3','i5', 'f5', 'c5', 'o5', 'a', 'ahat']}
        self.channels_a = channels_a  # size = n (first is the input size, followed by n-1 nchannels
        self.channels_r = channels_r  # size = n
        self.layer_size = len(channels_r)
        self.glob_filter_size = glob_filter_size
        self.output_mode = output_mode
        assert len(channels_a) == len(self.channels_r), 'channels in a and r should be equals, please check your arguments'
        for i in range(self.layer_size):
            for c in ['i1', 'f1', 'c1', 'o1']:
                act = 'tanh' if c == 'c' else 'hard_sigmoid'
                self.conv_layers[c].append(Conv2D(self.channels_r[i], 1, padding='same', activation=act))
            for c in ['i3', 'f3', 'c3', 'o3']:
                act = 'tanh' if c == 'c' else 'hard_sigmoid'
                self.conv_layers[c].append(Conv2D(self.channels_r[i], 3, padding='same', activation=act))
            for c in ['i5', 'f5', 'c5', 'o5']:
                act = 'tanh' if c == 'c' else 'hard_sigmoid'
                self.conv_layers[c].append(Conv2D(self.channels_r[i], 5, padding='same', activation=act))

            self.conv_layers['ahat'].append(Conv2D(self.channels_a[i], self.glob_filter_size, padding='same', activation='relu'))

        for i in range(1, self.layer_size):
            self.conv_layers['a'].append(Conv2D(self.channels_a[i], self.glob_filter_size, padding='same', activation='relu'))

        self.upsample = UpSampling2D()
        self.pool = MaxPooling2D()
        self.input_spec = [InputSpec(ndim=5)]

    def compute_output_shape(self, input_shape):
        if self.output_mode == 'prediction':
            out_shape = input_shape[2:]
        elif self.output_mode == 'error':
            out_shape = (self.layer_size,)
        if self.return_sequences:
            return (input_shape[0], input_shape[1]) + out_shape
        else:
            return (input_shape[0],) + out_shape

    def get_initial_state(self, x):
        input_shape = self.input_spec[0].shape
        init_nb_row = input_shape[-2]
        init_nb_col = input_shape[-1]
        base_initial_state = backend.zeros_like(x)
        for _ in range(2):
            base_initial_state = backend.sum(base_initial_state, axis=-1)
        base_initial_state = backend.sum(base_initial_state, axis=1)  # (samples, nb_channels)
        states_to_pass = ['r', 'c', 'e']
        initial_states = []
        nlayers_to_pass = {u: self.layer_size for u in states_to_pass}
        for u in states_to_pass:
            for l in range(nlayers_to_pass[u]):
                ds_factor = 2 ** l
                nb_row = init_nb_row // ds_factor
                nb_col = init_nb_col // ds_factor
                if u in ['r', 'c']:
                    stack_size = 3 * self.channels_r[l]
                elif u == 'e':
                    stack_size = 2 * self.channels_a[l]
                elif u == 'ahat':
                    stack_size = self.channels_a[l]
                output_size = stack_size * nb_row * nb_col  # flattened size

                reducer = backend.zeros((input_shape[-3], output_size))  # (nb_channels, output_size)
                initial_state = backend.dot(base_initial_state, reducer) # (samples, output_size)
                output_shp = (-1, stack_size, nb_row, nb_col)
                initial_state = backend.reshape(initial_state, output_shp)
                initial_states += [initial_state]
        return initial_states

    def build(self, input_shape):
        #recursively calling build on all its layers
        self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = []
        nb_row, nb_col = (input_shape[-2], input_shape[-1])
        for c in sorted(self.conv_layers.keys()):
            for l in range(len(self.conv_layers[c])):
                ds_factor = 2 ** l
                if c == 'ahat':
                    nb_channels = self.channels_r[l]*3
                elif c == 'a':
                    nb_channels = 2 * self.channels_a[l]
                else:
                    nb_channels = self.channels_a[l] * 2 + 3*self.channels_r[l]
                    if l < self.layer_size - 1:
                        nb_channels += self.channels_r[l + 1]*3
                in_shape = (input_shape[0], nb_channels, nb_row // ds_factor, nb_col // ds_factor)
                with backend.name_scope('layer_' + c + '_' + str(l)):
                    self.conv_layers[c][l].build(in_shape)
                self.trainable_weights += self.conv_layers[c][l].trainable_weights

        self.states = [None] * self.layer_size * 3

    def step(self, a, states):
        r_tm1 = states[:self.layer_size]
        c_tm1 = states[self.layer_size:2 * self.layer_size]
        e_tm1 = states[2 * self.layer_size:3 * self.layer_size]
        c = []
        r = []
        e = []
        for l in reversed(range(self.layer_size)):
            inputs = [r_tm1[l], e_tm1[l]]
            if l < self.layer_size - 1:
                inputs.append(r_up)
            inputs = backend.concatenate(inputs, axis=-3)
            f = backend.concatenate([self.conv_layers['f1'][l].call(inputs), self.conv_layers['f3'][l].call(inputs)],
                                    axis=-3)
            f = backend.concatenate([f, self.conv_layers['f5'][l].call(inputs)], axis=-3)
            i = backend.concatenate([self.conv_layers['i1'][l].call(inputs), self.conv_layers['i3'][l].call(inputs)],
                                    axis=-3)
            i = backend.concatenate([i, self.conv_layers['i5'][l].call(inputs)], axis=-3)
            o = backend.concatenate([self.conv_layers['o1'][l].call(inputs), self.conv_layers['o3'][l].call(inputs)],
                                    axis=-3)
            o = backend.concatenate([o, self.conv_layers['o5'][l].call(inputs)], axis=-3)
            g = backend.concatenate([self.conv_layers['c1'][l].call(inputs), self.conv_layers['c3'][l].call(inputs)],
                                    axis=-3)
            g = backend.concatenate([g, self.conv_layers['c5'][l].call(inputs)], axis=-3)
            _c = f * c_tm1[l] + i * g
            _r = o * activations.get('tanh')(_c)
            c.insert(0, _c)
            r.insert(0, _r)

            if l > 0:
                r_up = self.upsample.call(_r)

        for l in range(self.layer_size):
            ahat = self.conv_layers['ahat'][l].call(r[l])
            if l == 0:
                ahat = backend.minimum(ahat, 1.)
                frame_prediction = ahat

            # compute errors
            e_up = activations.get('relu')(ahat - a)
            e_down = activations.get('relu')(a - ahat)

            e.append(backend.concatenate((e_up, e_down), axis=-3))
            if(self.output_mode == 'error'):
                output  = e[l]
            output = ahat

            if l < self.layer_size - 1:
                a = self.conv_layers['a'][l].call(e[l])
                a = self.pool.call(a)  # target for next layer

        states = r + c + e
        if self.output_mode == 'prediction':
            output = frame_prediction
        else:
            for l in range(self.layer_size):
                layer_error = backend.mean(backend.batch_flatten(e[l]), axis=-1, keepdims=True)
                all_error = layer_error if l == 0 else backend.concatenate((all_error, layer_error), axis=-1)
            if self.output_mode == 'error':
                output = all_error
        return output, states

    def get_config(self):
        config = {'Channels in a': self.channels_a,
                  'Channels in r': self.channels_r,
                  'Global filter size': self.glob_filter_size}
        base_config = super(PredNetInceptionCustom, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))