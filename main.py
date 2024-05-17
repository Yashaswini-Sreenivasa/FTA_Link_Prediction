from src.generator import Node, FaultTree, FaultTreeGenerator, Converter, Generator
import matplotlib.pyplot as plt
import networkx as nx

# Params
    # Data Params
num_models = 1
    # Tree Params
    
start_id = 0
max_depth = 5
max_children = 3
min_children = 2


if __name__ == '__main__':

    Gen = Generator(start_id, max_depth, max_children, min_children, num_models)
    G = Gen.generate_data()
    
    pos = nx.planar_layout(G) 

    '''
    Displays node attributes on the graph
    node_labels = {node: f"ID: {data['id']}\nP: {data.get('failure_probability', 'N/A')}"  
                   for node, data in G.nodes(data=True)}
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=5000, node_color='lightblue', font_size=8)  
    '''

    #Displays node attributes on the console
    nx.draw(G, pos, node_size=50, node_color='lightblue', font_size=8)
    plt.show()
