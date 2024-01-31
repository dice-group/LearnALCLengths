import sys, os, json
import numpy as np, random


from celoe_clp.fast_concept_learner import CELOECLP
from ontolearn import KnowledgeBase
from ontolearn.refinement_operators import ExpressRefinement
from ontolearn.metrics import F1, Accuracy
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLNamedIndividual
from owlapy.model._iri import IRI
from util.data import Data

import sys, os

base_path = "./"

import argparse
if __name__=='__main__':
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        elif v.lower() in ['t', 'true', 'y', 'yes', '1']:
            return True
        elif v.lower() in ['f', 'false', 'n', 'no', '0']:
            return False
        else:
            raise ValueError('Invalid boolean value.')
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', type=str, default="carcinogenesis", help="Knowledge base name")
    parser.add_argument('--max_exec_time', type=int, default=120, help="The maximum execution time of CELOE-CLP")
    parser.add_argument('--iter_bound', type=int, default=100, help="The maximum number of search steps")
    parser.add_argument('--max_num_of_concepts_tested', type=int, default=30000, help="The maximum number of concepts to test during the search process")
    parser.add_argument('--terminate_on_goal', type=str2bool, const=True, default=True, nargs='?', help="Whether to terminate on goal")
    parser.add_argument('--best_only', type=str2bool, const=True, default=True, nargs='?', help="Whether to pick the best node at each step of the search")
    parser.add_argument('--calculate_min_max', type=str2bool, const=True, default=True, nargs='?', help="Whether to calculate min max horizontal expansion")
    parser.add_argument('--max_results', type=int, default=10, help="The maximum number of nodes in the queue at each step of the search process")
    args = parser.parse_args()
    
    KB = KnowledgeBase(path=f"{base_path}Datasets/{args.kb}/{args.kb}.owl")
    
    with open(base_path+f"Datasets/{args.kb}/Learning_problems/learning_problems.json", "r") as file_lp:
        learning_problems = json.load(file_lp)

    path_to_triples = base_path+f"Datasets/{args.kb}/Triples/"
    triples = Data({"path_to_triples":path_to_triples})

    with open("max_length.json") as file:
        max_length = json.load(file)
    as_classification = True
    rho = ExpressRefinement(KB)
    prefix = list(KB.individuals())[0].get_iri().as_str()
    prefix = prefix[:prefix.rfind("/")+1]

    num_classes = 1+max_length[args.kb]
    kwargs = {'knowledge_base': KB,
                 'learning_problem': None,
                 'refinement_operator': rho,
                 'quality_func': PosNegLPStandard,
                 'heuristic_func': CELOEHeuristic(),
                 'terminate_on_goal': args.terminate_on_goal,
                 'iter_bound': args.iter_bound,
                 'max_num_of_concepts_tested': args.max_num_of_concepts_tested,
                 'max_runtime': args.max_exec_time,
                 'max_results': args.max_results,
                 'best_only': args.best_only,
                 'calculate_min_max': args.calculate_min_max,
                 "learner_name":"GRU", "emb_model_name":"", "pretrained_embedding_path":"",
                 "pretrained_length_learner":f"Datasets/{args.kb}/Model_weights/GRU.pt",
                 "path_to_csv_embeddings":f"Embeddings/{args.kb}/ConEx_entity_embeddings.csv",
                 "learning_rate":0.003, "decay_rate":0, "path_to_triples":path_to_triples,
                 "random_seed":1, "embedding_dim":20, "num_entities":len(triples.entities),
                 "num_relations":len(triples.relations), "num_ex":1000, "input_dropout":0.0, 
                 "kernel_size":4, "num_of_output_channels":8, "feature_map_dropout":0.1,
                 "hidden_dropout":0.1, "rnn_n_layers":2,'rnn_hidden':100, 'input_size':41,
                 'linear_hidden':200, 'out_size':num_classes, 'dropout_prob': 0.1, 'num_units':500,
                 'seed':10, 'seq_len':1000,'kernel_w':5, 'kernel_h':11, 'stride_w':1, 'stride_h':7,
                 'conv_out':2040, 'mlp_n_layers':4, "as_classification":as_classification
         }
    
    max_num_lps = 100
    learning_problems = list(learning_problems.items())
    random.seed(1)
    random.shuffle(learning_problems)
    algo = CELOECLP(kwargs)
    results = {}
    print("#"*50)
    print("On {} KG".format(args.kb.capitalize()))
    print("#"*50)

    for i, (target_str, value) in enumerate(learning_problems):
        print(f"###Learning {target_str}###")
        pos = value['positive examples']
        neg = value['negative examples']
        pos = set(map(OWLNamedIndividual, map(IRI.create, map(lambda x: prefix+x, pos))))
        neg = set(map(OWLNamedIndividual, map(IRI.create, map(lambda x: prefix+x, neg))))
        lps = PosNegLPStandard(KB, pos, neg)
        Acc = Accuracy(lps) # Added to compute accuracy of the solution found
        algo.lp = lps
        algo.quality_func = F1(lps)
        algo.clp.load_pretrained()
        predicted_length = algo.clp.predict(pos, neg)
        algo.operator.max_child_length = predicted_length
        algo.clean()
        algo.fit()
        celoe_clp_results = algo.result_dict(target_str)
        solution = celoe_clp_results.pop('Prediction-Obj')
        for key in celoe_clp_results:
            results.setdefault(key, []).append(celoe_clp_results[key])
        _, acc = Acc.score(KB.individuals_set(solution))
        results.setdefault('Accuracy', []).append(acc)
        results.setdefault('Pred-Length', []).append(predicted_length)
        if i == max_num_lps-1:
            break

    avg_results = {}            
    for key in results:
        if not key in ["Learned Concept", "Prediction"]:
            avg_results.setdefault(key, {}).update({"mean": np.mean(results[key]), "std": np.std(results[key])})
    with open(base_path+f"Poster/{args.kb}/Results/concept_learning_results_clip.json", "w") as results_file:
        json.dump(results, results_file, ensure_ascii=False, indent=3)
          
    with open(base_path+f"Poster/{args.kb}/Results/concept_learning_avg_results_clip.json", "w") as avg_results_file:
        json.dump(avg_results, avg_results_file, indent=3)

    print()
    print("Avg results: ", avg_results)
    print()