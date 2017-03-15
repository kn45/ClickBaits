# ClickBaits

Mainly about short text classification with different model and corresponding features.

Use semi-supervised method to label data. ~100 more labeled data per iteration per coder.

Specified document in detail would be released later.

| id   | Model                      | Features                                 |
| :--- | :------------------------- | :--------------------------------------- |
| gbvc | GBDT                       | embedded sentence vector                 |
| gbvk | GBDT                       | embedded sentence vector + key words     |
| gblr | GBDT + LR                  | embedded sentence vector + key words     |
| rnnw | RNN(LSTM etc.)             | one-hot words with pre-trained embedding matrix |
| mrnn | multi-layer RNN(LSTM etc.) | one-hot words with pre-trained embedding matrix |
| rnnc | RNN(LSTM etc.)             | one-hot character without embedding      |
| cnnt | CNN(medium length text)    | text words                               |

