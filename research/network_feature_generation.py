import os
import json

import pandas as pd
from tqdm import tqdm
import networkx as nx
from node2vec import Node2Vec


FILE_WITH_ADDITIONAL_NETWORK_DATA = 'PATH'


def get_network_data(path=FILE_WITH_ADDITIONAL_NETWORK_DATA):
    bill_id_to_data = {}
    for subdir, dirs, files in tqdm(os.walk(path)):
        for file in files:
            f_path = os.path.join(subdir, file)
            if f_path.endswith('data.json'):
                with open(f_path, 'r') as f:
                    d = json.load(f)
                try:
                    bill_id = d['bill_id']
                    bill_id_to_data[bill_id] = d
                except:
                    pass
    return bill_id_to_data


def get_related_bills(bill_id, bill_info):
    if bill_id not in bill_info:
        return 'no_info'
    b = bill_info[bill_id].get('related_bills')
    if b:
        return b
    return 'empty'


def parse_related_bills(related_bills):
    if isinstance(related_bills, list):
        return len(related_bills)
    return 0


def get_bill_ids(related_bills):
    if isinstance(related_bills, list):
        return [l['bill_id'] for l in related_bills]
    return []


def assign_property(node_id, prop):
    return prop.get(node_id, 0)


def get_node_vector(b_id, model, embedding_size=16):
    try:
        vector = model.wv.get_vector(b_id)
        return vector
    except:
        return [0. for _ in range(embedding_size)]


def create_network(all_bills, bill_id_to_data):
    all_bills['related_bills'] = all_bills['b_id'].apply(lambda x: get_related_bills(x, bill_id_to_data))
    all_bills = all_bills[all_bills.related_bills != 'no_info']
    all_bills['related_bills_count'] = all_bills['related_bills'].apply(lambda x: parse_related_bills(x))
    all_bills['related_bills_list'] = all_bills['related_bills'].apply(lambda x: get_bill_ids(x))
    network = all_bills[['b_id', 'related_bills_list']]
    final_network = []
    for b_id, related_bills_list in network.values:
        for related_b in related_bills_list:
            final_network.append((b_id, related_b))
    final_network = pd.DataFrame(final_network, columns=['from', 'to'])
    G = nx.from_pandas_edgelist(final_network, source='from', target='to', create_using=nx.Graph())

    # Simple features
    centrality = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G)
    clustering = nx.clustering(G)
    page_rank = nx.algorithms.link_analysis.pagerank_alg.pagerank(G)
    betweenness = nx.betweenness_centrality(G, k=100)

    # Embeddings
    node2vec = Node2Vec(G, dimensions=16, walk_length=10, num_walks=5, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    all_bills = all_bills[['b_id', 'text', 'lobbied']]
    all_bills['centrality'] = all_bills.b_id.apply(lambda x: centrality.get(x, 0))
    all_bills['closeness'] = all_bills.b_id.apply(lambda x: closeness.get(x, 0))
    all_bills['clustering'] = all_bills.b_id.apply(lambda x: clustering.get(x, 0))
    all_bills['page_rank'] = all_bills.b_id.apply(lambda x: page_rank.get(x, 0))
    all_bills['betweenness'] = all_bills.b_id.apply(lambda x: betweenness.get(x, 0))
    all_bills['node2vec'] = all_bills.b_id.apply(lambda x: get_node_vector(x, model))
    for i in range(16):
        all_bills.drop([f'vector_{i}'], axis=1, inplace=True)
    vectors = all_bills['node2vec'].apply(pd.Series)
    all_bills = pd.concat([all_bills, vectors], axis=1)
    return all_bills


