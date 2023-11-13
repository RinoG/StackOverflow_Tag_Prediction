# StackOverflow Tag Prediction

As part of the Large Scale Machine Learning course, a tag proposal for Stack Overflow was made as a final project. 

The following dataset was used to fine tune BERT: [60k Stack Overflow Questions with Quality Rating](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate/?select=valid.csv)

I had problems with multi-tag predictions, so I only took the first tag. The input was the title of the question.

# 1. Model Training
This is the [Jupyter Notebook](https://github.com/RinoG/StackOverflow_Tag_Prediction/blob/main/model_training.ipynb)

And the models are stored [here](https://drive.google.com/drive/folders/1qSG9-jJ511Qy525_Gbm-LH5WdiLYGJwr)

I got the following metrics:

| Model                      | F1 Score | Precision | Recall |
|----------------------------|----------|-----------|--------|
| Baseline                   | 0.291    | 0.355     | 0.381  |
| BERT                       | 0.547    | 0.532     | 0.578  |
| BERT Pruning               | 0.534    | 0.519     | 0.573  |
| BERT Quantization          | 0.533    | 0.525     | 0.570  |
| BERT Prune and Quantization| 0.416    | 0.491     | 0.452  |

In terms of prediction speed the quantization showed best performance:

    Standard Model Average Time: 0.0341 seconds
    Pruned Model Average Time: 0.0341 seconds
    Quantized Model Average Time: 0.0183 seconds
    Pruned and Quantization Model Average Time: 0.0178 seconds

`BERT Quantization`  was used for the rest of the project.

# 2. Service deployment
Unfortunately, time is running out, so I did the service part quick and dirty. 

- download the [fine-tuned-model](https://drive.google.com/drive/folders/1qSG9-jJ511Qy525_Gbm-LH5WdiLYGJwr) and add the folder to the directory
- make sure packages from `requirements.txt` are installed
- run `server.py` in terminal like this: `python3 server.py`

Now you can send requests in python like this:

```python
import requests

url = "http://localhost:5000/predict"

data = {
    "title": "This is an example Title"
}
response = requests.post(url, json=data)

print("Response from API:", response.json())
```
output: `Response from API: {'predicted_tag': 'php'}`

## Performance Testing
Running `Apache JMeter` with following settings:
- Number of Threads (users): 10
- Ramp-up period (seconds): 5
- Loop Count: 100
- body: {"title": "This is an example title"}

I got this metrics:

| Metric            | Value     |
|-------------------|-----------|
| Label             | HTTP Request |
| # Samples         | 1501      |
| Average (ms)      | 128       |
| Min (ms)          | 0         |
| Max (ms)          | 257       |
| Std. Dev.         | 66.95     |
| Error %           | 0.000%    |
| Throughput        | 5.58946 requests/sec |
| Received KB/sec   | 1.1       |
| Sent KB/sec       | 1.2       |
| Avg. Bytes        | 202       |

