from . import create_all_graph_components


def create_keissler_graph(xy_grid, refinement_factor=2, merge_components=True):
    return create_all_graph_components(
        xy=xy_grid,
        merge_components=merge_components,
        m2m_connectivity="flat",
        m2m_connectivity_kwargs=dict(refinement_factor=refinement_factor),
        m2g_connectivity="nearest_neighbor",
    )
    
    

def create_graphcast_graph(xy_grid, refinement_factor=2, max_num_levels=None, merge_components=True):
    return create_all_graph_components(
        xy=xy_grid,
        m2m_connectivity="flat_multiscale",
        merge_components=merge_components,
        m2m_connectivity_kwargs=dict(refinement_factor=refinement_factor, max_num_levels=max_num_levels),
    )
    

def create_oscarsson_hierarchical_graph(xy_grid, merge_components=True):
    return create_all_graph_components(
        xy=xy_grid,
        m2m_connectivity="hierarchical",
        merge_components=merge_components,
    )