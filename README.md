# Tweet sentiment support phrase
Extract support phrases for sentiment labels. [Competition link](https://www.kaggle.com/c/tweet-sentiment-extraction/overview/kernels-requirements)

*Competition Last Date:* Jun 17, 2020

### Objective

In this project we have to find the supporting words which determine the polarity of the sentence.

**Approach 1** *We formulate the task as question answering problem: given a question and a context, we train a transformer model to find the answer in the text column (the context).*

*We have:*

*Question: sentiment column (positive or negative)* *Context: text column* *Answer: selected_text column*



**Approach 2** Text extraction using bert w/ sentiment: inference



### Evaluation Criteria

Jaccard score

> Jaccard Score is a measure of how similar/dissimilar two sets are. The higher the score, the more similar the two strings. The idea is to find the number of common tokens and divide it by the total number of unique tokens.



### Data

```
Training data shape:  (27486, 4)
Testing data shape:  (3535, 3)
```

**Features :** 	text, selected_text, sentiment

**e.g :** 	Oh! Good idea about putting them on ice cream, Good, positive



### Dependencies

1. [Hugging face tokenizer](https://github.com/huggingface/tokenizers)

   ```
   pip install tokenizers
   ```

2. Pytorch



### Competition rules

1. Submissions to this competition must be made through Kernels

2. GPU/CPU Kernel <= 3 hours run-time

3. No internet access enabled

4. External data, freely & publicly available, is allowed, including pre-trained models

5. Submission file must be named "submission.csv"

   

### References 

1. [Readme ideas]: https://www.kaggle.com/parulpandey/basic-preprocessing-and-eda