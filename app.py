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
        'fraud': 'brown',
        'harassment': 'pink'
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
        centrality_measures['pagerank'] = nx.pagerank(G)
    except:
        st.warning("Could not compute PageRank.")
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
        incident_type = random.choice(["hoax call", "spam", "threat", "suspicious", "fraud", "harassment"])
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
            ["hoax call", "spam", "threat", "suspicious", "fraud", "harassment"]
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

def plot_community_detection(G, resolved_entities, algorithm="louvain"):
    """Detect communities and visualize them."""
    try:
        if algorithm == "louvain":
            from community import community_louvain
            partition = community_louvain.best_partition(G.to_undirected())
        elif algorithm == "label_propagation":
            partition = {node: label for node, label in nx.label_propagation_communities(G.to_undirected())}
        elif algorithm == "girvan_newman":
            # For small to medium graphs
            if G.number_of_nodes() > 100:
                st.warning("Girvan-Newman algorithm is computationally expensive. Using Louvain instead.")
                from community import community_louvain
                partition = community_louvain.best_partition(G.to_undirected())
            else:
                comp = nx.community.girvan_newman(G.to_undirected())
                partition = {}
                for i, communities in enumerate(comp):
                    if i > 3:  # Limit to first few iterations
                        break
                    for j, community in enumerate(communities):
                        for node in community:
                            partition[node] = j
        else:
            # Default to Louvain
            from community import community_louvain
            partition = community_louvain.best_partition(G.to_undirected())
            
        # Create node trace for plotly
        node_x = []
        node_y = []
        node_labels = []
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            pos = nx.spring_layout(G, seed=42)
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(resolved_entities.get(node, str(node)))
            node_colors.append(partition.get(node, 0))
            node_sizes.append(G.degree(node) * 2 + 5)
        
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
            text=node_labels,
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
        
        return fig, partition
    
    except Exception as e:
        st.error(f"Error in community detection: {e}")
        return None, {}

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
        
        ### Getting Started:
        1. Choose to use sample data or upload your own CSV files in the sidebar
        2. Select which data layers to include in the analysis
        3. Explore the different tabs to analyze the network from various perspectives
        """)
    
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
                    st.success("Added new social media interaction!")
                elif dataset_type == "telecom" and telecom_df is not None:
                    telecom_df.loc[len(telecom_df)] = new_data
                    st.success("Added new telecom record!")
                elif dataset_type == "incident" and incident_df is not None:
                    incident_df.loc[len(incident_df)] = new_data
                    st.success("Added new incident report!")
                else:
                    st.error(f"The {dataset_type} dataset is not available.")
                
                # Update the graph
                st.info("Graph will be updated on the next refresh.")
                time.sleep(1)
                st.rerun()
    
    # Tab 4: Community Detection
    with tab4:
        st.header("Community Detection")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            algorithm = st.selectbox(
                "Community Detection Algorithm",
                ["louvain", "label_propagation", "girvan_newman"],
                key="community_algorithm"
            )
            
            st.info("Community detection helps identify clusters of densely connected nodes.")
            
            if st.button("Run Community Detection"):
                with st.spinner("Detecting communities..."):
                    community_fig, community_partition = plot_community_detection(G, resolved_entities, algorithm)
                    if community_fig:
                        st.session_state['community_fig'] = community_fig
                        st.session_state['community_partition'] = community_partition
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
            
            # Create dataframe of communities
            community_df = pd.DataFrame(
                [(node, resolved_entities.get(node, str(node)), community) 
                 for node, community in partition.items()],
                columns=["Node ID", "Node Name", "Community"]
            )
            
            # Group by community
            st.subheader("Communities")
            community_sizes = community_df.groupby('Community').size().reset_index(name='Size')
            
            # Bar chart of community sizes
            fig = px.bar(
                community_sizes,
                x='Community',
                y='Size',
                title='Community Sizes',
                labels={'Community': 'Community ID', 'Size': 'Number of Nodes'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show nodes in each community
            st.subheader("Nodes by Community")
            community_to_show = st.selectbox(
                "Select Community:",
                sorted(set(partition.values())),
                key="community_to_show"
            )
            
            nodes_in_community = community_df[community_df['Community'] == community_to_show]
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