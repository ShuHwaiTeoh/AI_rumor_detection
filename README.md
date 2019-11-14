# ECE 57000 course project - Implement Rumor Detection with Top-down Recursive Tree-structured neural networks
## Dependencies:

Please install the following python libraries:

numpy version 1.16.4<br/>
pytorch version 1.0.1<br/>
matplotlib version 3.1.0<br/>

## Usage
Run "model/Main.py" to reproduce the experiments<br/>
Run "model/compare.py" to polt the results in the result folder<br/>
Use variable "dataset" in "model/Function_from_original_author.py" to specified which dataset is used.<br/>
Uncomment line 22 "optimizer = ..." and comment out line 21 "optimizer = ..." to switch optimizer from ADAM to SGD.<br/>
Use function "plot3" in "model/compare.py" to plot the results of TheanoSGD model, PyTorchSFD model, and PyTorchADAM model.<br/>
Use function "plot2" in "model/compare.py" to plot the results of PyTorchADAM model tested with Twitter15 amd Twitter16 datasets.<br/>

## Files written by me

model/Main.py<br/>
model/TD_RvNN.py<br/>
model/compare.py<br/>

## Acknowledgements

The theory of this implementation is from the paper [Rumor Detection on Twitter with Tree-structured Recursive Neural Networks](https://www.aclweb.org/anthology/P18-1184/)

Following files are from the original author of the paper [majingCUHK](https://github.com/majingCUHK/Rumor_RvNN)<br/>
1. model/function_from_original_author.py:<br/>
	use to load the data for the model and evaluate result<br/>
2. Files in the nfold folder and resource folder:<br/> 
	datasets for training and testing model<br/>

## License
[MIT](https://choosealicense.com/licenses/mit/)

