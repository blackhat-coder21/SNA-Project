import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
from pyvis.network import Network
import tempfile
import os
from pathlib import Path
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz
from collections import defaultdict
import re
import random
from faker import Faker
import time
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# Set page config
st.set_page_config(layout="wide", page_title="Social Network Analysis Dashboard", page_icon="🔍")

# Initialize faker for demo data
fake = Faker()
random.seed(42)
Faker.seed(42)

def get_dataset_requirements():
    """
    Return information about required columns for each dataset when user opts not to use sample data.
    
    Returns:
    dict: Dictionary containing information about required columns for each dataset
    """
    return {
        # "users": {
        #     "description": "Basic user information for entity resolution across datasets",
        #     "required_columns": {
        #         "id": "Unique identifier for each user",
        #         "name": "Full name of the user"
        #     }
        # },
        "social_media": {
            "description": "Social media interactions between users",
            "required_columns": {
                "source": "User ID of the sender/poster",
                "source_name": "Name of the sender/poster",
                "target": "User ID of the receiver/mentioned user",
                "target_name": "Name of the receiver/mentioned user",
                "timestamp": "Date and time of the interaction (YYYY-MM-DD HH:MM:SS)",
                "message": "Content of the social media post/message",
                "platform": "Social media platform (e.g., Twitter, Facebook)",
                "is_public": "Boolean indicating if the post is public",
                "likes": "Number of likes/reactions received"
            }
        },
        "telecom_logs": {
            "description": "Communication logs between users",
            "required_columns": {
                "source": "User ID of the caller/sender",
                "source_name": "Name of the caller/sender",
                "target": "User ID of the receiver",
                "target_name": "Name of the receiver",
                "duration": "Duration of call in seconds",
                "timestamp": "Date and time of the call/message (YYYY-MM-DD HH:MM:SS)",
                "call_type": "Type of communication (voice, sms, video)",
                "status": "Call status (completed, missed, rejected, busy)",
                "location": "Location of the caller (optional)"
            }
        },
        "incident_reports": {
            "description": "Reported incidents involving users",
            "required_columns": {
                "source": "User ID of the reporter",
                "source_name": "Name of the reporter",
                "target": "User ID of the reported person",
                "target_name": "Name of the reported person",
                "incident_type": "Type of incident (fraud, threat, harassment, suspicious)",
                "location": "Location where the incident occurred",
                "timestamp": "Date and time of the incident (YYYY-MM-DD HH:MM:SS)",
                "severity": "Severity rating (1-5 scale)",
                "resolved": "Boolean indicating if the incident was resolved",
                "report_id": "Unique identifier for the incident report"
            }
        }
    }


# Initialize session state for tracking new data points
if 'new_data_points' not in st.session_state:
    st.session_state.new_data_points = {
        'social_media': [],
        'telecom': [],
        'incident': []
    }

# Helper Functions
def generate_html_report(graph, name="network_graph"):
    """Generate a PyVis HTML file from a NetworkX graph."""
    net = Network(height="600px", width="100%", directed=True, notebook=False)
    
    # Add nodes with properties
    for node, attrs in graph.nodes(data=True):
        size = attrs.get('value', 10)  # Default size
        color = attrs.get('color', '#97c2fc')  # Default color
        title = attrs.get('title', f"Node ID: {node}")
        label = attrs.get('label', str(node))
        net.add_node(node, size=size, color=color, title=title, label=label)
    
    # Add edges with properties
    for u, v, attrs in graph.edges(data=True):
        width = attrs.get('width', 1)
        color = attrs.get('color', '#848484')
        title = attrs.get('title', '')
        net.add_edge(u, v, width=width, color=color, title=title)
    
    # Configure physics
    net.toggle_physics(True)
    net.show_buttons(filter_=['physics'])
    
    # Save and return the HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        path = tmpfile.name
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    
    # Remove temp file
    os.unlink(path)
    return html_content

def get_download_link(html_content, filename="network_graph.html"):
    """Generate a download link for the HTML content."""
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download interactive graph</a>'
    return href

def fuzzy_confidence(name1, name2):
    """Return the fuzzy match confidence between two names as a value between 0 and 1."""
    score = fuzz.ratio(name1, name2) / 100  # Normalize to 0-1
    return round(score, 2)

def compute_entity_resolution(id_to_names):
    """For each user in id_to_names mapping, resolve names and compute overall confidence."""
    resolved_entities = {}
    confidence_scores = {}
    for user in id_to_names:
        names_in_datasets = list(id_to_names[user])
        # If only one name is observed, use it with full (1.0) confidence.
        if len(names_in_datasets) == 1:
            resolved_entities[user] = names_in_datasets[0]
            confidence_scores[user] = 1.0
        else:
            # Use the first name as the base resolved name
            resolved_name = names_in_datasets[0]
            confidences = []
            for other_name in names_in_datasets[1:]:
                conf = fuzzy_confidence(resolved_name, other_name)
                confidences.append(conf)
            overall_confidence = round(sum(confidences) / len(confidences) if confidences else 1.0, 2)
            resolved_entities[user] = resolved_name
            confidence_scores[user] = overall_confidence
    return resolved_entities, confidence_scores

def build_graph(dfs, layer_tags):
    """Build a directed graph from dataframes with corresponding layer tags."""
    G = nx.DiGraph()
    id_to_names = defaultdict(set)
    
    for df, layer_tag in zip(dfs, layer_tags):
        if df is not None:
            for _, row in df.iterrows():
                source = row['source']
                target = row['target']
                source_name = row.get('source_name', str(source))
                target_name = row.get('target_name', str(target))
                
                # Update name mappings
                id_to_names[source].add(source_name)
                id_to_names[target].add(target_name)
                
                # Add nodes if not present
                if source not in G.nodes:
                    G.add_node(source)
                if target not in G.nodes:
                    G.add_node(target)
                
                # Add edge with attributes
                attrs = {k: v for k, v in row.items() if k not in ['source', 'target', 'source_name', 'target_name']}
                attrs['layer'] = layer_tag
                
                # Check if this is a newly added edge
                is_new = False
                for new_point in st.session_state.new_data_points[layer_tag]:
                    if new_point.get('source') == source and new_point.get('target') == target:
                        is_new = True
                        break
                
                if is_new:
                    # Highlight newly added edges with different colors
                    if layer_tag == 'social_media':
                        attrs['color'] = '#FF5733'  # Orange-red for social media
                    elif layer_tag == 'telecom':
                        attrs['color'] = '#33FF57'  # Green for telecom
                    elif layer_tag == 'incident':
                        attrs['color'] = '#3357FF'  # Blue for incident
                    attrs['width'] = 3  # Make newer edges thicker
                    attrs['is_new'] = True
                
                G.add_edge(source, target, **attrs)
    
    # Compute entity resolution and confidence scores
    resolved_entities, confidence_scores = compute_entity_resolution(id_to_names)
    
    # Add these to the graph
    for node in G.nodes():
        if node in resolved_entities:
            G.nodes[node]['label'] = resolved_entities[node]
        if node in confidence_scores:
            G.nodes[node]['confidence'] = confidence_scores[node]
            G.nodes[node]['value'] = 5 + confidence_scores[node] * 10  # Size based on confidence
    
    return G, id_to_names, resolved_entities, confidence_scores


def add_incident_type_colors(G, incident_df):
    """Add color based on incident types to graph nodes."""
    color_map = {
        'hoax call': 'red',
        'spam': 'orange',
        'threat': 'purple',
        'suspicious': 'yellow',
    }
    
    if incident_df is not None:
        # Create a mapping of user_id to incident_type
        user_incident_types = {}
        for _, row in incident_df.iterrows():
            if 'incident_type' in row:
                if 'source' in row and pd.notnull(row['source']):
                    user_incident_types[row['source']] = row['incident_type']
                if 'target' in row and pd.notnull(row['target']):
                    user_incident_types[row['target']] = row['incident_type']
        
        # Apply colors to nodes based on incident type
        for node in G.nodes():
            if node in user_incident_types:
                incident_type = user_incident_types[node]
                G.nodes[node]['incident_type'] = incident_type
                G.nodes[node]['color'] = color_map.get(incident_type, 'gray')
                
                # Update title to include incident type
                title = G.nodes[node].get('title', f"Node ID: {node}")
                G.nodes[node]['title'] = f"{title}, Type: {incident_type}"

def compute_centrality_measures(G):
    """Compute various centrality measures for the graph."""
    centrality_measures = {}
    
    # Basic centrality measures
    centrality_measures['degree'] = nx.degree_centrality(G)
    centrality_measures['in_degree'] = nx.in_degree_centrality(G)
    centrality_measures['out_degree'] = nx.out_degree_centrality(G)
    
    try:
        centrality_measures['betweenness'] = nx.betweenness_centrality(G)
    except:
        st.warning("Could not compute betweenness centrality.")
        centrality_measures['betweenness'] = {node: 0 for node in G.nodes()}
    
    try:
        centrality_measures['closeness'] = nx.closeness_centrality(G)
    except:
        st.warning("Could not compute closeness centrality.")
        centrality_measures['closeness'] = {node: 0 for node in G.nodes()}
    
    try:
        centrality_measures['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        st.warning("Could not compute eigenvector centrality.")
        centrality_measures['eigenvector'] = {node: 0 for node in G.nodes()}
    
    try:
        # Modified PageRank calculation with parameters to handle disconnected components
        centrality_measures['pagerank'] = nx.pagerank(
            G,
            alpha=0.85,  # Damping parameter
            personalization=None,
            max_iter=100,  # Reduced from default for better convergence
            tol=1e-06,  # Relaxed tolerance
            nstart=None,
            weight='weight',
            dangling=None
        )
    except Exception as e:
        st.warning(f"Could not compute PageRank. Error: {str(e)}")
        # Provide a fallback pagerank value for all nodes
        centrality_measures['pagerank'] = {node: 0 for node in G.nodes()}
    
    return centrality_measures

def plot_centrality_distribution(centrality_data, measure_name):
    """Create a distribution plot for centrality measures."""
    values = list(centrality_data.values())
    fig = px.histogram(
        x=values,
        title=f"{measure_name.capitalize()} Centrality Distribution",
        labels={'x': f'{measure_name.capitalize()} Centrality', 'y': 'Count'},
        opacity=0.7,
        color_discrete_sequence=['#3366CC']
    )
    fig.update_layout(showlegend=False)
    return fig

def plot_top_nodes_by_centrality(centrality_data, measure_name, resolved_entities, top_n=10):
    """Create a bar chart for top nodes by centrality."""
    sorted_data = sorted(centrality_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    nodes = [resolved_entities.get(node, str(node)) for node, _ in sorted_data]
    values = [val for _, val in sorted_data]
    
    fig = px.bar(
        x=nodes,
        y=values,
        title=f"Top {top_n} Nodes by {measure_name.capitalize()} Centrality",
        labels={'x': 'Node', 'y': f'{measure_name.capitalize()} Centrality'},
        color=values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def generate_sample_data():
    """Generate sample data for demonstration."""
    # Generate users
    num_users = 50
    users = []
    for i in range(num_users):
        name = fake.name()
        user_id = f"user_{i+1}"
        users.append({"id": user_id, "name": name})
    
    users_df = pd.DataFrame(users)
    
    # Helper: get fuzzy variants of names
    def name_variant(name):
        if random.random() < 0.3:
            parts = name.split()
            return f"{parts[0][0]}. {parts[-1]}"  # e.g., J. Smith
        elif random.random() < 0.5:
            return name.replace("e", "a")  # small typo
        return name
    
    # Generate social media data
    social_data = []
    for _ in range(60):
        sender = random.choice(users)
        receiver = random.choice([u for u in users if u != sender])
        timestamp = fake.date_time_between(start_date='-1y', end_date='now').strftime('%Y-%m-%d %H:%M:%S')
        message = fake.sentence(nb_words=6)
        social_data.append([
            sender["id"], name_variant(sender["name"]),
            receiver["id"], name_variant(receiver["name"]),
            timestamp, message
        ])
    
    social_df = pd.DataFrame(
        social_data, 
        columns=["source", "source_name", "target", "target_name", "timestamp", "message"]
    )
    
    # Generate telecom logs
    telecom_data = []
    for _ in range(80):
        caller = random.choice(users)
        receiver = random.choice([u for u in users if u != caller])
        duration = random.randint(10, 600)
        timestamp = fake.date_time_between(start_date='-1y', end_date='now').strftime('%Y-%m-%d %H:%M:%S')
        call_type = random.choice(["voice", "sms", "video"])
        telecom_data.append([
            caller["id"], name_variant(caller["name"]),
            receiver["id"], name_variant(receiver["name"]),
            duration, timestamp, call_type
        ])
    
    telecom_df = pd.DataFrame(
        telecom_data, 
        columns=["source", "source_name", "target", "target_name", "duration", "timestamp", "call_type"]
    )
    
    # Generate incident reports
    incident_data = []
    for _ in range(30):
        reporter = random.choice(users)
        suspect = random.choice([u for u in users if u != reporter])
        # Reduced to 4 incident types
        incident_type = random.choice(["spam", "threat", "hoax call", "suspicious"])
        location = fake.city()
        timestamp = fake.date_time_between(start_date='-1y', end_date='now').strftime('%Y-%m-%d %H:%M:%S')
        incident_data.append([
            reporter["id"], name_variant(reporter["name"]),
            suspect["id"], name_variant(suspect["name"]),
            incident_type, location, timestamp
        ])
    
    incident_df = pd.DataFrame(
        incident_data, 
        columns=["source", "source_name", "target", "target_name", "incident_type", "location", "timestamp"]
    )
    
    return social_df, telecom_df, incident_df


def get_dataset_info():
    """
    Return formatted information about dataset requirements for display in the UI
    when user chooses not to use sample data.
    
    Returns:
    str: Markdown formatted text with dataset requirements
    """
    requirements = get_dataset_requirements()
    info_text = "## Dataset Requirements\n\n"
    info_text += "When uploading your own data, please ensure your files match these structures:\n\n"
    
    for dataset_name, info in requirements.items():
        info_text += f"### {dataset_name.upper()}\n"
        info_text += f"{info['description']}\n\n"
        info_text += "**Required columns:**\n"
        for col, desc in info['required_columns'].items():
            info_text += f"- **{col}**: {desc}\n"
        info_text += "\n"
    
    return info_text

def visualize_graph_by_layer(G, layer_name):
    """Create a subgraph containing only edges of a specific layer."""
    subgraph = nx.DiGraph()
    
    # Add all nodes to maintain node identity
    for node, attrs in G.nodes(data=True):
        subgraph.add_node(node, **attrs)
    
    # Add only edges from the specified layer
    for u, v, attrs in G.edges(data=True):
        if attrs.get('layer') == layer_name:
            subgraph.add_edge(u, v, **attrs)
    
    return subgraph

def create_new_data_point(source_id, source_name, target_id, target_name, dataset_type):
    """Create a new data point based on dataset type."""
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if dataset_type == "social_media":
        message = st.text_input("Message:", fake.sentence())
        return {
            "source": source_id,
            "source_name": source_name,
            "target": target_id,
            "target_name": target_name,
            "timestamp": timestamp,
            "message": message
        }
    
    elif dataset_type == "telecom":
        duration = st.number_input("Duration (seconds):", min_value=1, value=60)
        call_type = st.selectbox("Call Type:", ["voice", "sms", "video"])
        return {
            "source": source_id,
            "source_name": source_name,
            "target": target_id,
            "target_name": target_name,
            "timestamp": timestamp,
            "duration": duration,
            "call_type": call_type
        }
    
    elif dataset_type == "incident":
        incident_type = st.selectbox(
            "Incident Type:", 
            ["hoax call", "spam", "threat", "suspicious"]
        )
        location = st.text_input("Location:", fake.city())
        return {
            "source": source_id,
            "source_name": source_name,
            "target": target_id,
            "target_name": target_name,
            "timestamp": timestamp,
            "incident_type": incident_type,
            "location": location
        }
    
    return {}

def validate_csv_structure(df, required_columns, dataset_type):
    """Validate that uploaded CSV has required columns."""
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Error: The {dataset_type} CSV must contain the column '{col}'")
            return False
    return True

def add_edge_weights_by_frequency(G):
    """Add edge weights based on frequency of interactions."""
    edge_counts = defaultdict(int)
    
    # Count occurrences of each edge
    for u, v in G.edges():
        edge_counts[(u, v)] += 1
    
    # Normalize weights to be between 1 and 5
    if edge_counts:
        max_count = max(edge_counts.values())
        min_count = min(edge_counts.values())
        range_count = max_count - min_count if max_count > min_count else 1
        
        # Set edge weights
        for (u, v), count in edge_counts.items():
            if range_count > 0:
                normalized_weight = 1 + 4 * (count - min_count) / range_count
            else:
                normalized_weight = 1
            G[u][v]['width'] = normalized_weight
    
    return G

def plot_community_detection(G, resolved_entities, algorithm="louvain", datasets=None):
    """Detect communities and visualize them.
    
    Parameters:
    - G: NetworkX graph
    - resolved_entities: Dictionary mapping node IDs to resolved names
    - algorithm: Community detection algorithm to use
    - datasets: A dictionary containing the dataframes for different datasets
    """
    try:
        # Check if we need to create custom partitions based on node attributes
        custom_partition = {}
        
        if algorithm == "custom" and datasets:
            # For incident reports, partition by incident_type
            if 'incident' in datasets and datasets['incident'] is not None:
                for _, row in datasets['incident'].iterrows():
                    if 'incident_type' in row and pd.notnull(row['incident_type']):
                        if 'source' in row and pd.notnull(row['source']):
                            custom_partition[row['source']] = f"incident_{row['incident_type']}"
                        if 'target' in row and pd.notnull(row['target']):
                            custom_partition[row['target']] = f"incident_{row['incident_type']}"
            
            # For social media, partition by message content (since platform might not exist)
            # We'll use message content length as a proxy for different platforms
            if 'social_media' in datasets and datasets['social_media'] is not None:
                for _, row in datasets['social_media'].iterrows():
                    if 'message' in row and pd.notnull(row['message']):
                        # Classify messages by length as a proxy for platform
                        msg_length = len(str(row['message']))
                        platform_type = "short_msg" if msg_length < 50 else "long_msg"
                        
                        if 'source' in row and pd.notnull(row['source']):
                            custom_partition[row['source']] = f"social_{platform_type}"
                        if 'target' in row and pd.notnull(row['target']):
                            custom_partition[row['target']] = f"social_{platform_type}"
            
            # For telecom logs, partition by call_type
            if 'telecom' in datasets and datasets['telecom'] is not None:
                for _, row in datasets['telecom'].iterrows():
                    if 'call_type' in row and pd.notnull(row['call_type']):
                        if 'source' in row and pd.notnull(row['source']):
                            custom_partition[row['source']] = f"telecom_{row['call_type']}"
                        if 'target' in row and pd.notnull(row['target']):
                            custom_partition[row['target']] = f"telecom_{row['call_type']}"
        
        # Use custom partition if created and we're in custom mode, otherwise use standard algorithm
        if algorithm == "custom" and custom_partition:
            partition = custom_partition
            # Convert string partition values to integers for visualization
            unique_communities = list(set(partition.values()))
            community_mapping = {comm: i for i, comm in enumerate(unique_communities)}
            integer_partition = {node: community_mapping[comm] for node, comm in partition.items()}
            
            # Create a mapping to store the original community names
            community_names = {i: comm for comm, i in community_mapping.items()}
        else:
            # Use standard community detection algorithms with fallbacks
            if algorithm == "louvain":
                try:
                    # Try python-louvain package
                    from community import community_louvain
                    partition = community_louvain.best_partition(G.to_undirected())
                except ImportError:
                    # Fallback to networkx implementation
                    try:
                        from networkx.algorithms.community import greedy_modularity_communities
                        communities = list(greedy_modularity_communities(G.to_undirected()))
                        partition = {}
                        for i, comm in enumerate(communities):
                            for node in comm:
                                partition[node] = i
                    except:
                        # Last resort: just use degree as a proxy for communities
                        partition = {node: min(G.degree(node) // 2, 5) for node in G.nodes()}
                
                integer_partition = partition
                community_names = {i: f"Community {i}" for i in set(partition.values())}
                
            elif algorithm == "label_propagation":
                try:
                    # Try the correct function name
                    from networkx.algorithms.community import label_propagation_communities
                    communities = list(label_propagation_communities(G.to_undirected()))
                    partition = {}
                    for i, comm in enumerate(communities):
                        for node in comm:
                            partition[node] = i
                except ImportError or AttributeError:
                    # Fallback to a different community detection algorithm
                    try:
                        from networkx.algorithms.community import greedy_modularity_communities
                        communities = list(greedy_modularity_communities(G.to_undirected()))
                        partition = {}
                        for i, comm in enumerate(communities):
                            for node in comm:
                                partition[node] = i
                    except:
                        # Last resort: just use degree as a proxy for communities
                        partition = {node: min(G.degree(node) // 2, 5) for node in G.nodes()}
                
                integer_partition = partition
                community_names = {i: f"Community {i}" for i in set(partition.values())}
                
            elif algorithm == "girvan_newman":
                try:
                    # For small to medium graphs - use only first iteration to avoid computational issues
                    if G.number_of_nodes() > 100:
                        # Use degree centrality for large graphs
                        partition = {node: min(G.degree(node) // 2, 5) for node in G.nodes()}
                    else:
                        from networkx.algorithms.community import girvan_newman
                        comp = list(girvan_newman(G.to_undirected()))
                        if comp:  # Take just the first level of communities
                            communities = list(comp[0])
                            partition = {}
                            for i, community in enumerate(communities):
                                for node in community:
                                    partition[node] = i
                        else:
                            partition = {node: 0 for node in G.nodes()}
                except:
                    # Fallback to simpler approach
                    partition = {node: min(G.degree(node) // 2, 5) for node in G.nodes()}
                
                integer_partition = partition
                community_names = {i: f"Community {i}" for i in set(partition.values())}
            else:
                # Default fallback
                partition = {node: min(G.degree(node) // 2, 5) for node in G.nodes()}
                integer_partition = partition
                community_names = {i: f"Community {i}" for i in set(partition.values())}
            
        # Create node trace for plotly
        node_x = []
        node_y = []
        node_labels = []
        node_colors = []
        node_sizes = []
        node_hover_texts = []
        pos = nx.spring_layout(G, seed=42)
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_name = resolved_entities.get(node, str(node))
            node_labels.append(node_name)
            
            community_id = integer_partition.get(node, 0)
            node_colors.append(community_id)
            
            node_sizes.append(G.degree(node) * 2 + 5)
            
            # Create hover text with community info
            if algorithm == "custom" and node in partition:
                hover_text = f"Node: {node_name}<br>Community: {partition[node]}"
            else:
                comm_name = community_names.get(community_id, f"Community {community_id}")
                hover_text = f"Node: {node_name}<br>Community: {comm_name}"
            node_hover_texts.append(hover_text)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create the figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_hover_texts,
            marker=dict(
                color=node_colors,
                size=node_sizes,
                colorscale='Viridis',
                colorbar=dict(title='Community'),
                line=dict(width=1, color='#333')
            ),
            showlegend=False
        ))
        
        fig.update_layout(
            title=f'Community Detection ({algorithm})',
            showlegend=False,
            hovermode='closest',
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig, integer_partition, community_names
    
    except Exception as e:
        st.error(f"Error in community detection: {str(e)}")
        return None, {}, {}

def temporal_analysis(df, time_column="timestamp"):
    """Analyze temporal patterns in the data."""
    if df is None or df.empty or time_column not in df.columns:
        return None
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        try:
            df[time_column] = pd.to_datetime(df[time_column])
        except:
            st.error(f"Could not convert {time_column} to datetime format.")
            return None
    
    # Create time series of interactions
    df_ts = df.copy()
    df_ts['day'] = df_ts[time_column].dt.date
    df_ts['hour'] = df_ts[time_column].dt.hour
    df_ts['weekday'] = df_ts[time_column].dt.weekday
    
    # Activity by day
    daily_activity = df_ts.groupby('day').size().reset_index(name='count')
    
    # Activity by hour of day
    hourly_activity = df_ts.groupby('hour').size().reset_index(name='count')
    
    # Activity by day of week
    weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                  4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    weekday_activity = df_ts.groupby('weekday').size().reset_index(name='count')
    weekday_activity['weekday_name'] = weekday_activity['weekday'].map(weekday_map)
    
    return {
        'daily': daily_activity,
        'hourly': hourly_activity,
        'weekday': weekday_activity
    }

def plot_temporal_analysis(temporal_data):
    """Create plots for temporal analysis."""
    if not temporal_data:
        return None, None, None
    
    # Daily activity plot
    fig_daily = px.line(
        temporal_data['daily'], 
        x='day', 
        y='count',
        title='Activity by Day',
        labels={'day': 'Date', 'count': 'Number of Interactions'}
    )
    
    # Hourly activity plot
    fig_hourly = px.bar(
        temporal_data['hourly'], 
        x='hour', 
        y='count',
        title='Activity by Hour of Day',
        labels={'hour': 'Hour of Day', 'count': 'Number of Interactions'}
    )
    
    # Weekday activity plot
    fig_weekday = px.bar(
        temporal_data['weekday'], 
        x='weekday_name', 
        y='count',
        title='Activity by Day of Week',
        labels={'weekday_name': 'Day of Week', 'count': 'Number of Interactions'},
        category_orders={"weekday_name": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
    )
    
    return fig_daily, fig_hourly, fig_weekday

# UI Components
def sidebar_components():
    """Create sidebar components for data upload and settings."""
    st.sidebar.title("Data Sources")
    # Option to use sample data
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    # Store this choice in session state so we can access it in main()
    st.session_state['use_sample_data'] = use_sample_data
    # File uploads
    social_df = telecom_df = incident_df = None
    
    if use_sample_data:
        st.sidebar.info("Using generated sample data for demonstration.")
        social_df, telecom_df, incident_df = generate_sample_data()
    else:
        with st.sidebar.expander("Upload Social Media Data"):
            social_file = st.file_uploader("Social Media CSV", type="csv")
            if social_file:
                social_df = pd.read_csv(social_file)
                if not validate_csv_structure(social_df, ['source', 'target'], "social media"):
                    social_df = None
        
        with st.sidebar.expander("Upload Telecom Data"):
            telecom_file = st.file_uploader("Telecom Logs CSV", type="csv")
            if telecom_file:
                telecom_df = pd.read_csv(telecom_file)
                if not validate_csv_structure(telecom_df, ['source', 'target'], "telecom"):
                    telecom_df = None
        
        with st.sidebar.expander("Upload Incident Reports"):
            incident_file = st.file_uploader("Incident Reports CSV", type="csv")
            if incident_file:
                incident_df = pd.read_csv(incident_file)
                if not validate_csv_structure(incident_df, ['source', 'target'], "incident reports"):
                    incident_df = None
    
    # Select which datasets to include
    st.sidebar.subheader("Data Layers to Include")
    include_social = st.sidebar.checkbox("Social Media", value=True)
    include_telecom = st.sidebar.checkbox("Telecom Logs", value=True)
    include_incident = st.sidebar.checkbox("Incident Reports", value=True)
    
    # Process selection
    datasets = []
    layer_tags = []
    
    if include_social and social_df is not None:
        datasets.append(social_df)
        layer_tags.append('social_media')
    
    if include_telecom and telecom_df is not None:
        datasets.append(telecom_df)
        layer_tags.append('telecom')
    
    if include_incident and incident_df is not None:
        datasets.append(incident_df)
        layer_tags.append('incident')
    
    return datasets, layer_tags, social_df, telecom_df, incident_df

def main():
    """Main function to create the dashboard."""
    st.title("Social Network Analysis Dashboard 🔍")
    
    # Get data from sidebar
    datasets, layer_tags, social_df, telecom_df, incident_df = sidebar_components()
    
    # Intro text
    with st.expander("About this Dashboard", expanded=False):
        st.markdown("""
        This dashboard allows you to analyze complex social networks by visualizing relationships and calculating key network metrics.
        
        ### Features:
        - **Upload your own CSV data** or use sample data
        - **Visualize networks** with interactive graphs
        - **Compute centrality measures** to identify important nodes
        - **Add real-time data points** to see immediate changes
        - **Run community detection** algorithms
        - **Analyze temporal patterns** in the data
        - **Color-coded node categories** to easily distinguish data types:  
            - <span style="color:red"><strong>'hoax call'</strong>: red</span>  
            - <span style="color:orange"><strong>'spam'</strong>: orange</span>  
            - <span style="color:purple"><strong>'threat'</strong>: purple</span>  
            - <span style="color:gold"><strong>'suspicious'</strong>: yellow</span>  
        ---
        ### Getting Started:
        1. Choose to use sample data or upload your own CSV files in the sidebar
        2. Select which data layers to include in the analysis
        3. Explore the different tabs to analyze the network from various perspectives
        """, unsafe_allow_html=True)
    
    # Check if we're using sample data
    use_sample_data = st.session_state.get('use_sample_data', True)
    files_uploaded = social_df is not None or telecom_df is not None or incident_df is not None
    
    # Show dataset requirements when not using sample data and no files uploaded yet
    if not use_sample_data and not files_uploaded:
        st.header("Dataset Requirements")
        st.markdown(get_dataset_info())

    # Check if we have data to work with
    if not datasets:
        st.warning("Please select at least one dataset to include in the analysis.")
        return
    
    # Build network graph
    G, id_to_names, resolved_entities, confidence_scores = build_graph(datasets, layer_tags)
    
    # Add incident type colors
    add_incident_type_colors(G, incident_df)
    
    # Add edge weights
    G = add_edge_weights_by_frequency(G)
    
    # Compute centrality measures
    centrality_measures = compute_centrality_measures(G)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕸️ Network Visualization", 
        "📊 Centrality Analysis", 
        "➕ Add Data Points",
        "👥 Community Detection",
        "⏱️ Temporal Analysis"
    ])
    
    # Tab 1: Network Visualization
    with tab1:
        st.header("Network Visualization")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            view_type = st.radio(
                "View Type",
                ["Unified Network", "By Layer"],
                key="view_type"
            )
            
            if view_type == "By Layer":
                selected_layer = st.selectbox(
                    "Select Layer",
                    [tag for tag in layer_tags],
                    key="selected_layer"
                )
                
                layer_graph = visualize_graph_by_layer(G, selected_layer)
                html_content = generate_html_report(layer_graph, f"{selected_layer}_graph")
                
                # Network stats
                st.subheader("Network Statistics:")
                st.write(f"Nodes: {layer_graph.number_of_nodes()}")
                st.write(f"Edges: {layer_graph.number_of_edges()}")
                
                if layer_graph.number_of_nodes() > 0:
                    density = nx.density(layer_graph)
                    st.write(f"Density: {density:.4f}")
                
                # Download link
                st.markdown(get_download_link(html_content, f"{selected_layer}_network.html"), unsafe_allow_html=True)
            else:
                html_content = generate_html_report(G, "unified_graph")
                
                # Network stats
                st.subheader("Network Statistics:")
                st.write(f"Nodes: {G.number_of_nodes()}")
                st.write(f"Edges: {G.number_of_edges()}")
                
                if G.number_of_nodes() > 0:
                    density = nx.density(G)
                    st.write(f"Density: {density:.4f}")
                    
                    n_components = nx.number_weakly_connected_components(G)
                    st.write(f"Weakly Connected Components: {n_components}")
                    
                    if nx.number_strongly_connected_components(G) > 0:
                        avg_clustering = nx.average_clustering(G.to_undirected())
                        st.write(f"Average Clustering: {avg_clustering:.4f}")
                    
                # Download link
                st.markdown(get_download_link(html_content, "unified_network.html"), unsafe_allow_html=True)
        
        with col2:
            st.components.v1.html(html_content, height=600)
        
        # Display top connected nodes
        # Display top connected nodes
        st.subheader("Most Connected Nodes")
        
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
        top_nodes_df = pd.DataFrame(
            [(resolved_entities.get(node, str(node)), degree) for node, degree in top_nodes],
            columns=["Node", "Connections"]
        )
        
        st.dataframe(top_nodes_df)
        
        # Display entity resolution confidence
        st.subheader("Entity Resolution")
        
        low_confidence_entities = {
            user: (resolved_entities[user], score)
            for user, score in confidence_scores.items()
            if score < 0.8
        }
        
        if low_confidence_entities:
            st.warning("Some users have low confidence entity resolution.")
            
            low_conf_df = pd.DataFrame(
                [(user, name, score) for user, (name, score) in low_confidence_entities.items()],
                columns=["User ID", "Resolved Name", "Confidence"]
            )
            
            st.dataframe(low_conf_df)
            
            # Option to view all names linked to a user
            selected_user = st.selectbox(
                "View All Names for User:",
                options=list(low_confidence_entities.keys())
            )
            
            if selected_user in id_to_names:
                st.write("Names associated with this user ID:")
                for name in id_to_names[selected_user]:
                    st.write(f"- {name}")
        else:
            st.success("All entity resolutions have high confidence.")
    
    # Tab 2: Centrality Analysis
    with tab2:
        st.header("Centrality Analysis")
        
        # Select centrality measure
        selected_measure = st.selectbox(
            "Select Centrality Measure",
            ["degree", "in_degree", "out_degree", "betweenness", "closeness", "eigenvector", "pagerank"],
            key="centrality_measure"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plot
            fig_dist = plot_centrality_distribution(
                centrality_measures[selected_measure], 
                selected_measure
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Top nodes by centrality
            fig_top = plot_top_nodes_by_centrality(
                centrality_measures[selected_measure],
                selected_measure,
                resolved_entities,
                top_n=10
            )
            st.plotly_chart(fig_top, use_container_width=True)
        
        # Show centrality data in a table
        st.subheader("Centrality Data")
        centrality_df = pd.DataFrame.from_dict(
            centrality_measures[selected_measure], orient='index', columns=[selected_measure]
        ).reset_index()
        centrality_df.columns = ['Node ID', 'Centrality Value']
        centrality_df['Node Name'] = centrality_df['Node ID'].map(resolved_entities)
        centrality_df = centrality_df.sort_values('Centrality Value', ascending=False)
        
        st.dataframe(centrality_df)
        
        # Download centrality data
        csv = centrality_df.to_csv(index=False)
        st.download_button(
            "Download Centrality Data",
            csv,
            f"{selected_measure}_centrality.csv",
            "text/csv",
            key="download_centrality"
        )
    
    # Tab 3: Add Data Points
    with tab3:
        st.header("Add Data Points")
        
        # Select dataset to add data to
        dataset_type = st.selectbox(
            "Select Dataset Type",
            ["social_media", "telecom", "incident"],
            key="add_dataset_type"
        )
        
        # Select source and target nodes
        source_options = list(resolved_entities.keys())
        source_id = st.selectbox("Source Node:", source_options, key="source_node")
        source_name = resolved_entities.get(source_id, str(source_id))
        
        target_options = [node for node in source_options if node != source_id]
        target_id = st.selectbox("Target Node:", target_options, key="target_node")
        target_name = resolved_entities.get(target_id, str(target_id))
        
        # Create form for new data point
        with st.form("new_data_point"):
            st.subheader(f"Add New {dataset_type.capitalize()} Data Point")
            
            new_data = create_new_data_point(
                source_id, source_name,
                target_id, target_name,
                dataset_type
            )
            
            submit_button = st.form_submit_button("Add Data Point")
            
            if submit_button and new_data:
                # Add to appropriate dataframe
                if dataset_type == "social_media" and social_df is not None:
                    social_df.loc[len(social_df)] = new_data
                    st.session_state.new_data_points['social_media'].append(new_data)
                    st.success("Added new social media interaction!")
                elif dataset_type == "telecom" and telecom_df is not None:
                    telecom_df.loc[len(telecom_df)] = new_data
                    st.session_state.new_data_points['telecom'].append(new_data)
                    st.success("Added new telecom record!")
                elif dataset_type == "incident" and incident_df is not None:
                    incident_df.loc[len(incident_df)] = new_data
                    st.session_state.new_data_points['incident'].append(new_data)
                    st.success("Added new incident report!")
                else:
                    st.error(f"The {dataset_type} dataset is not available.")
        
        # Display newly added data points
        st.subheader("Recently Added Data Points")
        
        added_tabs = st.tabs(["Social Media", "Telecom", "Incident"])
        
        with added_tabs[0]:
            if st.session_state.new_data_points['social_media']:
                social_added_df = pd.DataFrame(st.session_state.new_data_points['social_media'])
                st.dataframe(social_added_df, use_container_width=True)
                st.info("New social media connections are highlighted in orange-red in the network visualization.")
            else:
                st.info("No social media data points added yet.")
        
        with added_tabs[1]:
            if st.session_state.new_data_points['telecom']:
                telecom_added_df = pd.DataFrame(st.session_state.new_data_points['telecom'])
                st.dataframe(telecom_added_df, use_container_width=True)
                st.info("New telecom connections are highlighted in green in the network visualization.")
            else:
                st.info("No telecom data points added yet.")
        
        with added_tabs[2]:
            if st.session_state.new_data_points['incident']:
                incident_added_df = pd.DataFrame(st.session_state.new_data_points['incident'])
                st.dataframe(incident_added_df, use_container_width=True)
                st.info("New incident connections are highlighted in blue in the network visualization.")
            else:
                st.info("No incident data points added yet.")
        
        # Display updated network visualization
        if any(len(points) > 0 for points in st.session_state.new_data_points.values()):
            st.subheader("Updated Network Visualization")
            
            # Rebuild the graph to include new data points
            G, id_to_names, resolved_entities, confidence_scores = build_graph(datasets, layer_tags)
            add_incident_type_colors(G, incident_df)
            G = add_edge_weights_by_frequency(G)
            
            # Generate and display the updated visualization
            html_content = generate_html_report(G, "updated_graph")
            st.components.v1.html(html_content, height=600)
            
            # Download link for updated graph
            st.markdown(get_download_link(html_content, "updated_network.html"), unsafe_allow_html=True)
            
            # Option to clear all added data point highlighting
            if st.button("Clear Highlighting of Added Points"):
                st.session_state.new_data_points = {
                    'social_media': [],
                    'telecom': [],
                    'incident': []
                }
                st.success("Cleared highlighting of added data points.")
                st.rerun()
            
    # Tab 4: Community Detection
    with tab4:
        st.header("Community Detection")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            algorithm = st.selectbox(
                "Community Detection Algorithm",
                ["custom", "louvain", "label_propagation", "girvan_newman"],
                key="community_algorithm"
            )
            
            st.info("Community detection helps identify clusters of densely connected nodes.")
            if algorithm == "custom":
                st.info("Custom mode clusters nodes based on:\n- Incident Type for incident reports\n- Platform for social media\n- Call Type for telecom logs")
            
            if st.button("Run Community Detection"):
                with st.spinner("Detecting communities..."):
                    # Create a dictionary of datasets
                    dataset_dict = {
                        'social_media': social_df,
                        'telecom': telecom_df,
                        'incident': incident_df
                    }
                    
                    # Pass the datasets to the function regardless of algorithm
                    # The function will only use them for custom algorithm
                    community_fig, community_partition, community_names = plot_community_detection(
                        G, resolved_entities, algorithm, dataset_dict
                    )
                    
                    if community_fig:
                        st.session_state['community_fig'] = community_fig
                        st.session_state['community_partition'] = community_partition
                        st.session_state['community_names'] = community_names
                    else:
                        st.error("Failed to detect communities.")
        
        with col2:
            if 'community_fig' in st.session_state:
                st.plotly_chart(st.session_state['community_fig'], use_container_width=True)
            else:
                st.write("Run community detection to visualize clusters.")
        
        # Show community data if available
        if 'community_partition' in st.session_state and st.session_state['community_partition']:
            partition = st.session_state['community_partition']
            community_names = st.session_state['community_names']
            
            # Create dataframe of communities
            community_data = []
            for node, comm_id in partition.items():
                comm_name = community_names.get(comm_id, f"Community {comm_id}")
                community_data.append({
                    "Node ID": node,
                    "Node Name": resolved_entities.get(node, str(node)),
                    "Community ID": comm_id,
                    "Community Name": comm_name
                })
            
            community_df = pd.DataFrame(community_data)
            
            # Group by community
            st.subheader("Communities")
            community_sizes = community_df.groupby(['Community ID', 'Community Name']).size().reset_index(name='Size')
            
            # Bar chart of community sizes
            fig = px.bar(
                community_sizes,
                x='Community ID',
                y='Size',
                title='Community Sizes',
                labels={'Community ID': 'Community', 'Size': 'Number of Nodes'},
                hover_data=['Community Name'],
                color='Community Name'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show nodes in each community
            st.subheader("Nodes by Community")
            
            # Format function to display community names in the selectbox
            def format_community(comm_id):
                return community_names.get(comm_id, f"Community {comm_id}")
            
            community_to_show = st.selectbox(
                "Select Community:",
                sorted(set(partition.values())),
                format_func=format_community,
                key="community_to_show"
            )
            
            nodes_in_community = community_df[community_df['Community ID'] == community_to_show]
            st.dataframe(nodes_in_community)
    
    # Tab 5: Temporal Analysis
    with tab5:
        st.header("Temporal Analysis")
        
        # Choose which dataset to analyze
        dataset_for_temporal = st.selectbox(
            "Select Dataset for Temporal Analysis",
            ["social_media", "telecom", "incident"],
            key="temporal_dataset"
        )
        
        selected_df = None
        if dataset_for_temporal == "social_media" and social_df is not None:
            selected_df = social_df
        elif dataset_for_temporal == "telecom" and telecom_df is not None:
            selected_df = telecom_df
        elif dataset_for_temporal == "incident" and incident_df is not None:
            selected_df = incident_df
        
        if selected_df is not None:
            # Perform temporal analysis
            temporal_data = temporal_analysis(selected_df)
            
            if temporal_data:
                fig_daily, fig_hourly, fig_weekday = plot_temporal_analysis(temporal_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(fig_daily, use_container_width=True)
                    st.plotly_chart(fig_weekday, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig_hourly, use_container_width=True)
                    
                    # Heatmap of activity
                    if 'hour' in selected_df.columns and 'weekday' in selected_df.columns:
                        st.subheader("Activity Heatmap: Hour vs Weekday")
                        
                        # Create pivot table
                        heatmap_data = selected_df.pivot_table(
                            index='weekday', 
                            columns='hour', 
                            aggfunc='count', 
                            fill_value=0
                        )
                        
                        # Plot heatmap
                        fig = px.imshow(
                            heatmap_data, 
                            labels=dict(x="Hour of Day", y="Day of Week", color="Activity"),
                            x=list(range(24)),
                            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No temporal data available for {dataset_for_temporal} dataset.")
        else:
            st.warning(f"The {dataset_for_temporal} dataset is not available.")
    
    # Credits and additional information
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        """
        This dashboard was created for network analysis and visualization.
        
        It supports multi-layer network analysis with entity resolution.
        """
    )

if __name__ == "__main__":
    main()