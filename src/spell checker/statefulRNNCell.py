import tensorflow as tf


class StatefulRNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, inner_cell):
        super(StatefulRNNCell, self).__init__()
        self._inner_cell = inner_cell

    @property
    def state_size(self):
        return self._inner_cell.state_size

    @property
    def output_size(self):
        return self._inner_cell.state_size

    def call(self, input, *args, **kwargs):
        output, next_state = self._inner_cell(input, *args, **kwargs)
        emit_output = next_state
        return emit_output, next_state
