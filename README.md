**FTA lINK PREDICTION**

FTA Link Prediction is a project focused on predicting missing links in large Fault Trees or multiple Fault Trees using a Graph Convolutional Network (GCN). 
This repository includes scripts for generating fault tree datasets and implementing the GCN model for link prediction.

Repository Structure

main.py: Script to build and preprocess the dataset of fault trees according to specific requirements.
generator.py: Utility script used alongside main.py for generating fault tree structures.

_**Supervised Learning Models:**_

GCN.py: Implementation of the Graph Convolutional Network (GCN) model to predict missing links in the fault trees. 

GAT.py: Implementation of the Graph Attention Network (GAT) model to predict missing links in the fault trees.

GraphSAGE: Implementation of the Graph Sample and Aggregation (GraphSAGE) model to predict missing links in the fault trees.

_**Reinforcement Learning Model:**_

Rl.py: Implementation of the Graph Convolutional Network layers and training the model as an agent to predict missing links in the fault trees. 
