from src.generator import Node, FaultTree, FaultTreeGenerator, Converter, Generator
import matplotlib.pyplot as plt
import networkx as nx

# Params
    # Data Params
num_models = 4
    # Tree Params
    
start_id = 0
max_depth = 3
max_children = 3
min_children = 2


if __name__ == '__main__':

    Gen = Generator(start_id, max_depth, max_children, min_children, num_models)
    G = Gen.generate_data()

    nx.draw(G)
    plt.show()
