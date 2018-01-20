# DataCenterJobSchedulingSolution

## Requirements:

python 2.7

python packages:

```bash
tensorflow >= 0.12.0rc0
tensorlayer >= 1.6.1
numpy >= 1.11.2 
matplotlib >= 1.3.1
sklearn >= 0.18.2
```

## Code Structure
```bash
/src: sources code
	/src/entity: tend to use for furture simulation
	/src/agent.py: class for agent
	/src/config.py: store all configuration 
	/src/environment.py: handle all training data
	/visualization.py: viusal results
/log: log file 
/dataset: self generated dataset
	/dataset/fakeDataGenerator.py: for generating some samples 
```
