Experimentation with Gated Recurrent Units (GRUs)

From https://blog.floydhub.com/gru-with-pytorch/:
" The Gated Recurrent Unit (GRU) is the younger sibling of the more popular Long Short-Term Memory (LSTM) network, and also a type of Recurrent Neural Network (RNN). Just like its sibling, GRUs are able to effectively retain long-term dependencies in sequential data. And additionally, they can address the “short-term memory” issue plaguing vanilla RNNs. "

GRU = "variant of the RNN architecture, that uses gating mechanisms to control and manage the flow of information between cells in the neural network. "

![](https://blog.floydhub.com/content/images/2019/07/image17-1.jpg)

" The structure of the GRU allows it to adaptively capture dependencies from large sequences of data without discarding information from earlier parts of the sequence. This is achieved through its gating units, similar to the ones in LSTMs, which solve the vanishing/exploding gradient problem of traditional RNNs. These gates are responsible for regulating the information to be kept or discarded at each time step. "

" The GRU cell contains only two gates: the **Update gate** and the **Reset gate**. Just like the gates in LSTMs, these gates in the GRU are trained to selectively filter out any irrelevant information while keeping what’s useful. These gates are essentially vectors containing values between 0 to 1 which will be multiplied with the input data and/or hidden state. A 0 value in the gate vectors indicates that the corresponding data in the input or hidden state is unimportant and will, therefore, return as a zero. On the other hand, a 1 value in the gate vector means that the corresponding data is important and will be used. "

![](https://blog.floydhub.com/content/images/2019/07/image14.jpg)

