# Evaluation

MP-NRGAN is used for neuron segmentation. The segmentation results should be fed to a tracer to get the tracing results, such as NGPST, APP2, and so on. To evaluate the tracing results, you should put the tracing results and the corresponding groundtruth in the folder. And run
```
python evaluation.py
```
