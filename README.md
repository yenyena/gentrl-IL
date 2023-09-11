# gentrl-IL
Extending the GENTRL model with imitation learning.

## Workflow:
You can check how data was acquired and preprocessed in data/data-preprocessing.ipynb. <br/>
The edited version of the GENTRL, which contains all the imitation learning functions, is in gentrl_v2 folder. <br/>
The model is trained by running pretrain.ipynb, which produces the organised latent space, and train_rl_il.ipynb, which trains the reinforcement or imitation learning agents.<br/>
After training, the model can be evaluated by running the last three notebooks. actives_model.ipynb contains the definition and training function for the model that predicts whether a SMILES string represent a molecule active or inactive on the target. sampling.ipynb produces a sample of 1000 SMILES from a given model. evaluate.ipynb calculates training metrics for a given sample and model (activity prediction, validity, uniqueness, novelty, QED...).
