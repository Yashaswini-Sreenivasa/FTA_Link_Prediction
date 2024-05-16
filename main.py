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

    # Generator 
    # FT, id = FaultTreeGenerator.generate_fault_tree(start_id, max_depth, max_children, min_children)


    # G = Converter.convert_to_graph(FT)

    # FT = FaultTree(start_id, min_children, max_children, max_depth)
    # for node in FT.nodes:
    #     print(node.id, node.node_type)

    Gen = Generator(start_id, max_depth, max_children, min_children, num_models)
    G = Gen.generate_data()

    nx.draw(G)
    plt.show()
