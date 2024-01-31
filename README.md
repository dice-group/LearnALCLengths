# LearnALCLengths
This repository contains our implementation of concept length predictors in the ALC description logic.

## Installation

- Clone this repository:
```
https://github.com/dice-group/LearnALCLengths.git
```
- Install Anaconda3, then all required librairies by executing the following commands (Linux):

1. ```conda create -n clip python==3.11.5 && conda activate clip ```
2. ```pip install -r requirements.txt ```
3. ```git clone https://github.com/dice-group/Ontolearn.git && cd Ontolearn && git checkout 0.5.4 && pip install -e .```

- Download DL-Learner-1.4.0 from [github](https://github.com/SmartDataAnalytics/DL-Learner/releases) and extract it into this repository (cloned above)

- Clone DLFoil and DLFocl [dlfoil](https://bitbucket.org/grizzo001/dl-foil.git), [dlfocl](https://bitbucket.org/grizzo001/dlfocl.git), and extract the two repositories into `LearnALCLengths/`

- Install Java (version 8+) and Apache Maven (Only necessary for running DL-Learner and DL-Foil/DL-Focl)

## Reproducing the reported results

### Datasets (necessary for running the algorithms)

- Download [datasets](https://hobbitdata.informatik.uni-leipzig.de/CLIP/Datasets-CLIP.zip) and extract the zip file into `LearnALCLengths/` and rename the folder as Datasets

### CLIP (our method)

*Open a terminal and navigate into /reproduce_results/ ``` cd LearnALCLengths/reproduce_results/```
- Reproduce CLIP concept learning results on all KBs ``` sh reproduce_celoe_clp_experiment_all_kbs.sh```
- Reproduce the training of concept length predictors ``` sh reproduce_training_clp_on_all_kbs.sh```
- Furthermore, one can train concept length predictors on a single knowledge base as follows  ``` python reproduce_training_length_predictors_K_kb.py```, where ```K``` is one of carcinogenesis, mutagenesis, semantic_bible or vicodi. Use -h to see more training options (example ```python reproduce_training_length_predictors_carcinogenesis_kb.py -h ```).

### CELOE, ELTL, OCEL from DL-Learner

*Open a terminal and navigate into /other_learning_systems/scripts ``` cd LearnALCLengths/dllearner/scripts```
- Reproduce concept learning results on knowledge base K for algorithm Algo ``` python reproduce_dllearner_experiment.py --learning_systems Algo --knowledge_bases K```
- To reproduce the results for multiple algorithms on multiple knowledge bases, use the schema ``` python reproduce_dllearner_experiment.py --learning_systems Algo1 Algo2... --knowledge_bases K1 K2...```

Note that ```Algo``` is one of celoe, ocel or eltl, and ```K``` is one of carcinogenesis, mutagenesis, semantic_bible or vicodi (all lower cased)

### DLFoil and DLFocl

*For DLFoil, open a terminal and navigate into /dl-foil/DLFoil2* ``` cd LearnALCLengths/dl-foil/DLFoil2```
- Run ```mvn clean install```
- Open a different terminal and run the following ```python LearnALCLengths/generators/generate_dlfoil_config_all_kbs.py```
- Now execute the following in the first terminal (in LearnALCLengths/dl-foil/DLFoil2): ```mvn -e exec:java -Dexec.mainClass=it.uniba.di.lacam.ml.DLFoilTest -Dexec.args=K_config.xml >> ../dlfoil_out_K.txt```, where `K` is one of carcinogenesis, mutagenesis, semantic_bible or vicodi.

Note that DLFoil fails to solve our learning problems as it gets stuck on the refinement of certain partial descriptions.

*We could not run DLFocl.* 

The authors did not provide sufficient documentation to run  their algorithm; the documentation is [here](https://bitbucket.org/grizzo001/dlfocl.git)


### Statistical Test

*Open a terminal and navigate into /reproduce_results/* ``` cd LearnALCLengths/reproduce_results/```
- Run Wilcoxon statistical test on concept learning results `All Algos vs CLIP`: ``` sh run_statistical_test_on_all_kbs.sh```

### Use your own data

- Add your data into Datasets: it should be a folder containing a file formatted as RDF/XML or OWL/XML and should have the same name as the folder.

- Navigate into /generators and run ```python train_data/generate_training_data.py --kb your_folder_name```, use -h to see more options. The generated file Data.json under ```your_folder_name/Train_data/``` should serve for training concept length predictors, see example scripts in ```/reproduce_results/train_clp/```.

- Similarly, learning problems can be generated using one of the example files in generators/learning_problems/ (replace folder names by your folder name)

- Navigate into /Embeddings/Compute-Embeddings/ and run the following to embed your knowledge base: ```python run_script.py --path_dataset_folder your_folder_name```

- Train concept length predictors by preparing and running your python file ``` reproduce_training_length_predictors_K_kb.py ``` following examples in ```/reproduce_results/train_clp/```.

- Finally, prepare a script (see examples in ```/reproduce_results/celoe_clp/```) and run CLIP on your data. 


## Acknowledgement 
We based our implementation on the open source implementation of [ontolearn](https://docs--ontolearn-docs-dice-group.netlify.app/). We would like to thank the Ontolearn team for the readable codebase.