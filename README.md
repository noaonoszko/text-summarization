# Text summarization using machine learning
Project in DAT450 - Machine Learning for Natural Language Processing

## Introduction
Automatic text summarization is the task of producing a concise version of a text, preserving the underlying principal information. There are two main ways to do this: Either by generating text from scratch, or by picking sentences from the article. In this work, we present two different methods for text summarization, using the latter approach in a supervised setting. The first method mimics the model presented in a research paper, using recurrent neural networks in a reinforcement learning setting as a measure to optimize a domain specific evaluation metric, the ROUGE score. This model is trained on the CNN/DailyMail dataset with ground truth summaries written by journalists. The second model is a simple recurrent neural network. This model was trained on articles from the Cornell Newsroom with ground truth summaries consisting of sentences from the article. The reinforcement learning model did not generalize, likely due to a bug. The simple model however showed good results, outperforming the strong LEAD-3 baseline. This success can partly be credited to the fact that the ground truth summaries contain sentences from the article.

Below is a schematic figure of the architecture of the reinforcement learning based model.
<p align="center">
  <img src="https://github.com/noaonoszko/text-summarization/blob/main/data/rl_model.png" width="600" title="hover text">
</p>


## Results
The simple RNN model outperformed the LEAD-3 baseline. However, it produces similar summaries since the first three sentences are very frequent as shown in the right figure.
<p align="center">
  <img src="https://github.com/noaonoszko/text-summarization/blob/main/data/LSTM_rouge.png" width="350" title="hover text">
    <img src="https://github.com/noaonoszko/text-summarization/blob/main/data/Leading_s1.png" width="350" title="hover text">
</p>

Below is a sample output summary and the corresponding target summary. The sentence number is specified in parenthesis. For example (0 1 2) means that the first three sentences from the input text are used. The sentences in bold font are common for the output and target summaries. Here, 2/3 sentences are the same.

Output:  (0 2 3)  
**Japan's central bank downgrades assessments of exports, output.** U.S. crude oil futures claw back some ground lost overnight. **TOKYO, Sept 15 (Reuters) - Asian shares struggled on Tuesday as caution reigned ahead of this week's U.S. Federal Reserve decision on interest rates, while the yen edged higher after the Bank of Japan refrained from any new policy steps.**

Target: (0 3 4)  
**Japan's central bank downgrades assessments of exports, output.  TOKYO, Sept 15 (Reuters) - Asian shares struggled on Tuesday as caution reigned ahead of this week's U.S. Federal Reserve decision on interest rates, while the yen edged higher after the Bank of Japan refrained from any new policy steps.** MSCI's broadest index of Asia-Pacific shares outside Japan erased early gains and fell 0.5 percent, taking its cue from slumping Chinese shares. 

The reinforcement learning model did not manage to generalize as can be seen by the below training graph.
<p align="center">
  <img src="https://github.com/noaonoszko/text-summarization/blob/main/data/rl_rouge_scores11.png" width="350" title="hover text">
</p>

## User guide

### Dependencies
To run the code, some packages are needed. You can create a conda environment with the required dependencies by running
`conda env create -f environment.yml`.

### Running the Code
The last version of the reinforcement learning based model is in the `noa` branch and the last version of the RNN model is in the `gustav` branch. The former is run by executing the `main.py` script with (or without) flags that are described in the code. The latter is run by executing the `src_stacked_lstm.py` script.

# License
MIT License

Copyright (c) 2020 Ahmed Groshar, Gustav Molander, Noa Onoszko 

