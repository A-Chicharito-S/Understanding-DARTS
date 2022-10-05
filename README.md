# Understanding-DARTS
this repo is the commented version (only the cnn folder is annotated, and the rnn folder follows similar logic) of code from: [here](https://github.com/quark0/darts), the implementation for the paper DARTS: Differentiable Architecture Search.

For citations and details on training etc., please refer to [here](https://github.com/quark0/darts).

To quickly understand the code (take the cnn folder as an example), we can simply devide them into two parts: 

1. **code with \'\_search\' suffix**; 2. **w/o \'\_search\' suffix**.

For part 1 (with suffix), they are used for searching the optimal architecture (alpha), and for part 2 (w/o suffix), they will use the optimal architecture searched by part 1, and **ONLY** learn the weights under such architecture.
