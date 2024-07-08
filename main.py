from generator import Node, FaultTree, FaultTreeGenerator, Converter, Generator
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Params
num_models = 5
start_id = 0
max_depth = 5
max_children = 5
min_children = 4

def get_graph_structure(graph):
    return nx.weisfeiler_lehman_graph_hash(graph)

if __name__ == '__main__':
    Gen = Generator(start_id, max_depth, max_children, min_children, num_models)
    G = Gen.generate_data()

    # Convert each unique model to CSV format
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append([u, v])

    nodes_data = []
    for node, data in G.nodes(data=True):
        node_info = data.copy()
        node_info['id'] = node
        nodes_data.append(node_info)

    edges_df = pd.DataFrame(edges_data, columns=["Source", "Target"])
    nodes_df = pd.DataFrame(nodes_data)

    csv_edges_filename = f"Edges.csv"
    csv_nodes_filename = f"Nodes.csv"

    edges_df.to_csv(csv_edges_filename, index=False)
    nodes_df.to_csv(csv_nodes_filename, index=False)

    pos = nx.planar_layout(G)

    '''
    Displays node attributes on the graph
    node_labels = {node: f"G: {data['gate_type']}"  
                   for node, data in G.nodes(data=True)}
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=500, node_color='lightblue', font_size=8)  
    '''

    # Displays node attributes on the console
    nx.draw(G, pos, node_size=50, node_color='lightblue', font_size=8)
    plt.show()
