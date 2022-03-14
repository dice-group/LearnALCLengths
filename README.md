# LearnALCLengths
This repository contains our implementation of concept length predictors in the ALC description logic.

## Installation

- Clone this repository:
```
https://github.com/dice-group/LearnALCLengths.git
```
- Install Anaconda3, then all required librairies by running the following:
```
conda env create -f environment.yml
```
A conda environment (clip) will be created. Next activate the environment:
``` conda activate clip```

- Download DL-Learner-1.4.0 from [github](https://github.com/SmartDataAnalytics/DL-Learner/releases) and extract it into Method (cloned above)

- Clone DLFoil and DLFocl [dlfoil](https://bitbucket.org/grizzo001/dl-foil.git), [dlfocl](https://bitbucket.org/grizzo001/dlfocl.git), and extract the two repositories into LearnLengths/Method

- Install Java (version 8+) and Apache Maven (Only necessary for running DL-Learner and DL-Foil/DL-Focl)

## Reproducing the reported results

### Datasets (necessary for running the algorithms)

- Download [datasets](https://hobbitdata.informatik.uni-leipzig.de/CLIP/Datasets-CLIP.zip) and extract the zip file into Method and rename the folder as Datasets

### CLIP (our method)

*Open a terminal and navigate into Method/reproduce_results/* ``` cd LearnLengths/Method/reproduce_results/```
- Reproduce CLIP concept learning results on all KBs ``` sh reproduce_celoe_clp_experiment_all_kbs.sh```
- Reproduce the training of concept length predictors ``` sh reproduce_training_clp_on_all_kbs.sh```
- Furthermore, one can train concept length predictors on a single knowledge base as follows  ``` python reproduce_training_length_predictors_K_kb```, where ```K``` is one of carcinogenesis, mutagenesis, semantic_bible or vicodi. Use -h to see more training options (example ```python reproduce_training_length_predictors_carcinogenesis_kb -h ```).

### CELOE, ELTL, OCEL from DL-Learner

*Open a terminal and navigate into Method/other_learning_systems/scripts* ``` cd LearnLengths/Method/dllearner/scripts```
- Reproduce concept learning results on knowledge base K for algorithm Algo ``` python reproduce_dllearner_experiment.py --learning_systems Algo --knowledge_bases K```
- To reproduce the results for multiple algorithms on multiple knowledge bases, use the schema ``` python reproduce_dllearner_experiment.py --learning_systems Algo1 Algo2... --knowledge_bases K1 K2...```

Note that ```Algo``` is one of celoe, ocel or eltl, and ```K``` is one of carcinogenesis, mutagenesis, semantic_bible or vicodi

### DLFoil and DLFocl

*For DLFoil, open a terminal and navigate into Method/dl-foil/DLFoil2* ``` cd LearnLengths/Method/dl-foil/DLFoil2```
- Run ```mvn clean install```
- Open a different terminal and run the following ```python LearnLengths/Method/generators/generate_dlfoil_config_all_kbs.py```
- Now execute the following in the first terminal (in LearnLengths/Method/dl-foil/DLFoil2): ```mvn -e exec:java -Dexec.mainClass=it.uniba.di.lacam.ml.DLFoilTest -Dexec.args=kb_config.xml >> ../dlfoil_out_kb.txt```, where `kb` is one of carcinogenesis, mutagenesis, semantic_bible or vicodi.

Note that DLFoil fails to solve our learning problems as it gets stuck on the refinement of certain partial descriptions.

*We could not run DLFocl.* 

The authors did not provide sufficient documentation to run  their algorithm; the documentation is [here](https://bitbucket.org/grizzo001/dlfocl.git)


### Statistical Test

*Open a terminal and navigate into Method/reproduce_results/* ``` cd LearnLengths/Method/reproduce_results/```
- Run Wilcoxon statistical test on concept learning results `All Algos vs CLIP`: ``` sh run_statistical_test_on_all_kbs.sh```

### Use your own data

- Add your data into Datasets: it should be a folder containing a file formatted as RDF/XML or OWL/XML and should have the same name as the folder.

- Navigate into Method/generators and run ```python train_data/generate_training_data.py --kb your_folder_name```, use -h to see more options. The generated file Data.json under ```your_folder_name/Train_data/``` should serve for training concept length predictors, see example scripts in ```Method/reproduce_results/train_clp/```.

- Similarly, learning problems can be generated using one of the example files in generators/learning_problems/ (replace folder names by your folder name)

- Navigate into Method/Embeddings/Compute-Embeddings/ and run the following to embed your knowledge base: ```python run_script.py --path_dataset_folder your_folder_name```

- Train concept length predictors by preparing and running your python file ``` reproduce_training_length_predictors_K_kb.py ``` following examples in ```Method/reproduce_results/train_clp/```.

- Finally, prepare a script (see examples in ```Method/reproduce_results/celoe_clp/```) and run CLIP on your data. 


## Acknowledgement 
We based our implementation on the open source implementation of [ontolearn](https://docs--ontolearn-docs-dice-group.netlify.app/). We would like to thank the Ontolearn team for the readable codebase.