# StackOverflow Tag Prediction

As part of the Large Scale Machine Learning course, a tag proposal for Stack Overflow was made as a final project. 

The following dataset was used to fine tune BERT: [60k Stack Overflow Questions with Quality Rating](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate/?select=valid.csv)

I had problems with multi-tag predictions, so I only took the first tag. The input was the title of the question.

# 1. Model Training
Models are stored [here](https://drive.google.com/drive/folders/1qSG9-jJ511Qy525_Gbm-LH5WdiLYGJwr)

I got the following metrics:

| Model                      | F1 Score | Precision | Recall |
|----------------------------|----------|-----------|--------|
| Baseline                   | 0.291    | 0.355     | 0.381  |
| BERT                       | 0.547    | 0.532     | 0.578  |
| BERT Pruning               | 0.534    | 0.519     | 0.573  |
| BERT Quantization          | 0.533    | 0.525     | 0.570  |
| BERT Prune and Quantization| 0.416    | 0.491     | 0.452  |

In terms of prediction speed the quantizised showed best performance:

    Standard Model Average Time: 0.0341 seconds
    Pruned Model Average Time: 0.0341 seconds
    Quantized Model Average Time: 0.0183 seconds
    Pruned and Quantization Model Average Time: 0.0178 seconds

For the further project the `BERT Quantization` was taken.
`BERT Quantization`  was used for the rest of the project.

# 2. Service deployment
