CS5787 HW2 Deep Learning with Prof. Hadar Elor - Sean Hardesty Lewis (shl225)
# Performing Language Modeling with LSTMs and GRUs on the Penn Tree Bank Dataset
## Overview
This project implements several variants of the "small" model as described in "Recurrent Neural Network Regularization" by Zaremba et al. for token prediction from the Penn Tree Bank dataset. The variants include:

* LSTM based network without dropout
* LSTM based network with dropout
* GRU based network without dropout
* GRU based network with dropout

The goal is to compare the perplexity of these techniques and achieve below 125 validation perplexity without dropout, and below 100 with it.

### Convergence Graphs

#### LSTM-based Network without Dropout
**Learning Rate:** Starting at 5.0, adjusted using a scheduler based on validation perplexity.

**Dropout Probability:** 0.0

<img src="https://github.com/user-attachments/assets/33c7b58e-7080-4d62-a3d7-6b0b8eea7241">

*Figure 1: LSTM without Dropout: Train and Validation Perplexity over 200 Epochs*

This graph illustrates how the training and validation perplexities decrease over 200 epochs for the LSTM model without dropout. The training perplexity decreases significantly, reaching around 44.93, while the validation perplexity decreases to approximately 141.88 but does not go below 125.

#### LSTM-based Network with Dropout
**Learning Rate:** Starting at 5.0, adjusted using a scheduler based on validation perplexity.

**Dropout Probability:** 0.25

<img src="https://github.com/user-attachments/assets/4f4d7138-2e64-4aa6-9262-650a5f889a7d">

*Figure 2: LSTM with Dropout (0.25): Train and Validation Perplexity over 200 Epochs*

This graph shows the training and validation perplexities for the LSTM model with a dropout of 0.25. The training perplexity decreases to about 65.99, and the validation perplexity reaches around 100.97 but does not fall below 100, even after extensive training and hyperparameter adjustments.

#### GRU-based Network without Dropout
**Learning Rate:** Starting at 5.0, adjusted using a scheduler based on validation perplexity.

**Dropout Probability:** 0.0

<img src="https://github.com/user-attachments/assets/df3473f1-5f6e-4adf-bd55-0beb9c6d643d">

*Figure 3: GRU without Dropout: Train and Validation Perplexity over 200 Epochs*

This graph presents the perplexities for the GRU model without dropout. The training perplexity decreases to approximately 35.62, but the validation perplexity flatlines around 153.75, remaining above 125 despite extended training.

#### GRU-based Network with Dropout
**Learning Rate:** Starting at 5.0, adjusted using a scheduler based on validation perplexity.

**Dropout Probability:** 0.28

<img src="https://github.com/user-attachments/assets/e35b004e-1ae2-4c8f-8686-58b9dacd02d3">

*Figure 4: GRU with Dropout (0.28): Train and Validation Perplexity over 200 Epochs*

This graph depicts the perplexities for the GRU model with a dropout of 0.28. The training perplexity reduces to about 73.28, and the validation perplexity reaches approximately 105.10. Despite adjusting hyperparameters, the validation perplexity did not drop below 100.

### Summary of Results

Below is a table summarizing the final training perplexities, the minimum validation perplexities, and the test perplexity achieved at the minimum validation achieved by each model. The models are selected based on their lowest validation perplexity.

| Model Type | Dropout | Final Train Perplexity | Min Validation Perplexity | Test Perplexity at Min |
|------------|---------|------------------------|---------------------------|------------------------|
| LSTM       | 0.0     | 44.93                  | 141.88                    | 139.78                 |
| GRU        | 0.0     | 35.62                  | 153.75                    | 96.92                  |
| LSTM       | 0.25    | 65.99                  | 100.97                    | 157.20                 |
| GRU        | 0.28    | 73.28                  | 105.10                    | 101.85                 |

### Conclusions

From the experiments conducted, several observations can be made:

1. **Impact of Dropout:** Introducing dropout improved the validation perplexity for both LSTM and GRU models. For the LSTM, the validation perplexity decreased from approximately 141.88 (without dropout) to 100.97 (with dropout). Similarly, for the GRU, it decreased from about 153.75 to 105.10.
2. **Training vs. Validation Perplexity:** While the training perplexities continued to decrease significantly over the epochs, the validation perplexities plateaued after a certain point. This indicates that the models were overfitting to the training data, especially noticeable in the models without dropout.
3. **Difficulty Achieving Target Perplexities:** Despite extensive training and adjusting hyperparameters such as dropout rates and learning rates, I was unable to reduce the validation perplexity to below 100 for models with dropout and below 125 for models without dropout. This suggests that either with my current architecture and settings, there might be limitations in the model's capacity to generalize better on the validation set.
4. **GRU vs. LSTM Performance:** The LSTM models performed better on the validation set compared to the GRU models in terms of achieving lower perplexities. This could be due to the LSTM's ability to capture longer dependencies more effectively than GRUs in this context.

### Training Instructions
To train the models with my hyperparameters, use the following commands:

#### LSTM without Dropout:
```bash
config = TrainConfig(cell_type='LSTM', dropout=0.0, epochs=200)
```

#### LSTM with Dropout:
```bash
config = TrainConfig(cell_type='LSTM', dropout=0.25, epochs=200)
```

#### GRU without Dropout:
```bash
config = TrainConfig(cell_type='GRU', dropout=0.0, epochs=200)
```

#### GRU with Dropout:
```bash
config = TrainConfig(cell_type='GRU', dropout=0.28, epochs=200)
```

### Saving the Weights
To save the weights of the trained models, use the following commands:

#### All Models:
```python
if valid_perplexity < best_valid_perplexity:
    best_valid_perplexity = valid_perplexity
    torch.save(model.state_dict(), f'best_model_{config.cell_type}_dropout{config.dropout}.pth')
```

### Testing with Saved Weights
To test the models with saved weights, use the following commands:

#### LSTM without Dropout:
```python
model = RNNModel(vocab_size=vocab_size, hidden_size=200, num_layers=2, dropout=0.0, model_type='LSTM')
model.load_state_dict(torch.load('best_model_LSTM_dropout0.0.pth'))
model.to(device)
```

#### LSTM with Dropout:
```python
model = RNNModel(vocab_size=vocab_size, hidden_size=200, num_layers=2, dropout=0.25, model_type='LSTM')
model.load_state_dict(torch.load('best_model_LSTM_dropout0.25.pth'))
model.to(device)
```

#### GRU without Dropout:
```python
model = RNNModel(vocab_size=vocab_size, hidden_size=200, num_layers=2, dropout=0.0, model_type='GRU')
model.load_state_dict(torch.load('best_model_GRU_dropout0.0.pth'))
model.to(device)
```

#### GRU with Dropout:
```python
model = RNNModel(vocab_size=vocab_size, hidden_size=200, num_layers=2, dropout=0.28, model_type='GRU')
model.load_state_dict(torch.load('best_model_GRU_dropout0.28.pth'))
model.to(device)
```

## References

1. **Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling**  
   Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio.  
   *arXiv preprint arXiv:1412.3555*, 2014.  
   [arXiv:1412.3555](https://arxiv.org/abs/1412.3555)

2. **A Comparison of LSTM and GRU Networks for Learning Symbolic Sequences**  
   Roberto Cahuantzi, Xinye Chen, Stefan Güttel.  
   *arXiv preprint arXiv:2107.02248*, 2021.  
   [arXiv:2107.02248](https://arxiv.org/abs/2107.02248)

3. **Building a Large Annotated Corpus of English: The Penn Treebank**  
   Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz.  
   *Computational Linguistics*, 19(2):313–330, 1993.

