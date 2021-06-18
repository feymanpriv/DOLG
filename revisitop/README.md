# Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking


## Datasets

Roxford5k: (https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)
Rpairs6k: (https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)
R1M: (http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/)


## Python

Tested with Python 3.7 on CentOS.

### Extract features


### Evaluate results

Example script that describes how to evaluate according to the revisited annotation and the three protocol setups:
```
>> python3 example_evaluate
```
The final output should look like this (depending on the selected ```test_dataset```):
```
>> roxford5k: mAP E: 84.81, M: 64.67, H: 38.47
>> roxford5k: mP@k[ 1  5 10] E: [97.06 92.06 86.49], M: [97.14 90.67 84.67], H: [81.43 63.   53.  ]
```
or
```
>> rparis6k: mAP E: 92.12, M: 77.2, H: 56.32
>> rparis6k: mP@k[ 1  5 10] E: [100.    97.14  96.14], M: [100.    98.86  98.14], H: [94.29 90.29 89.14]
```


### Reference

```
(https://github.com/filipradenovic/revisitop)
```


## Related publication

```
@inproceedings{RITAC18,
 author = {Radenovi\'{c}, F. and Iscen, A. and Tolias, G. and Avrithis, Y. and Chum, O.},
 title = {Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking},
 booktitle = {CVPR},
 year = {2018}
}
```
