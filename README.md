# CrossFall
A cross-dataset deep learning-based classifier for people fall detection and identification.

Note: this repository is currently under development.

<!-- TABLE OF CONTENTS -->
## Table of Contents
* [Abstract](#abstract)
* [Authors](#authors)
* [License](#license)
* [Citation](#citation)

## Abstract
### Background and Objective.
Fall detection is an important problem for vulnerable sectors of the population such as elderly people, who frequently live alone. Note that a fall can be very dangerous for them if they cannot ask for help. Hence, in those situations, an automatic system that detected and informed to emergency services about the fall and subject identity could help to save lives. This way, they would know not only when but also who to help. Thus, our objective is to develop a new approach, based on deep learning, for fall detection and people identification that can be used in different datasets without any fine-tuning of the model parameters.

### Methods. 
We present a dataset-independent deep learning-based model that, by employing a multi-task learning approach, uses raw inertial information as input to solve simultaneously two tasks: fall detection and subject identification. By this way, our approach is able to automatically learn the best representations without any constraint introduced by the pre-processed features.

### Results.
Our cross-dataset classifier is able to detect falls with more than a 98% of accuracy in four datasets recorded under different conditions (i.e. accelerometer device, sampling rate, sequence length, age of the subjects, etc.). Moreover, the number of false positives is very low -- on average less than 1.6% -- establishing a new state-of-the-art. Finally, our classifier is also capable of  correctly identifying people with an average accuracy of 79.6%.

### Conclusions.
The presented approach performs both tasks (fall detection and people  identification) by using a single model and achieving real-time execution. The obtained results allow us to assert that a single model can be used for both fall detection and people identification under different conditions, easing its real implementation, as it is not necessary to train the model for new subjects.


## Authors

* **Rubén Delgado Escaño** - [rubende](https://github.com/rubende)
* **Francisco Castro Payán** - [fcastro](https://github.com/fcastro)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Citation

If you use this code in your research, please cite:

```
@article{delgado2020cross,
  title={A cross-dataset deep learning-based classifier for people fall detection and identification},
  author={Delgado-Escaño, Rubén and Castro, Francisco M and Cózar, Julián R and Marín-Jiménez, Manuel J and Guil, Nicolás and Casilari, Eduardo},
  journal={Computer methods and programs in biomedicine},
  volume={184},
  pages={105265},
  year={2020},
  publisher={Elsevier}
}
```

## Acknowledgments

* We thank the reviewers for their helpful comments.
