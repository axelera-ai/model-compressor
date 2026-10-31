import networkx as nx

import model_compressor.passes.adapt.base as base
import model_compressor.passes.adapt.coalesce_subgraphs as coalesce_subgraphs


def validate_meta_graph(graph: base.nx.Graph, meta_graph: coalesce_subgraphs._MetaGraph) -> None:
    semantics_to_subgraph = {}
    for edge in graph.edges:
        subgraph = base.nx.get_subgraph(graph, edge)
        semantics_to_subgraph[subgraph.semantics] = subgraph
    edges = {edge for edge in graph.edges}
    assert nx.number_weakly_connected_components(meta_graph) == 1
    assert nx.is_directed_acyclic_graph(meta_graph)
    for meta_node in meta_graph.nodes:
        subgraph = semantics_to_subgraph[meta_node]
        subgraph_ = base.nx.Graph()
        subgraph_.add_edges_from(subgraph.edges)
        for edge in subgraph_.edges:
            edges.remove(edge)
        assert nx.number_weakly_connected_components(subgraph_) == 1
        assert nx.is_directed_acyclic_graph(subgraph_)
