# Multi-Source Network Analysis Project

This project demonstrates the application of Social Network Analysis (SNA) techniques to analyze and visualize interconnected data from multiple sources. The system ingests data from social media interactions, telecom logs, and incident reports to build a comprehensive network representation for analysis.

## Overview

The project simulates a scenario where multiple data streams need to be combined for analysis:
- Social media interactions (sender-receiver relationships)
- Telecommunication logs (caller-receiver relationships)
- Incident reports (reporter-suspect relationships)

It then uses entity resolution techniques to merge these disparate sources into a unified network graph that can be analyzed and visualized.

## Features

- **Multi-source Data Generation**: Creates synthetic datasets to simulate real-world data sources
- **Entity Resolution**: Uses fuzzy matching to identify and link entities across different data sources
- **Network Visualization**: Provides multiple visualization options using Pyvis, Plotly, and NetworkX
- **Graph Analytics**: Calculates key network metrics including centrality measures and community detection
- **Dynamic Updates**: Supports adding new data points to the existing network
- **Dashboard**: You can visualize layered graphs

## Requirements

```
pandas
numpy
faker
networkx
fuzzywuzzy
pyvis
plotly
matplotlib
seaborn
streamlit
```

## Installation

```bash
pip install pandas numpy faker networkx fuzzywuzzy pyvis plotly matplotlib seaborn stramlit
```

## RUN

```bash
streamlit run main_visualization_app.py
```

## Visualization Options

The project includes three visualization methods:

1. **Pyvis**: Interactive HTML-based visualization with detailed hover information
2. **Plotly**: Interactive visualization with additional customization options
3. **Matplotlib/Seaborn**: Static visualizations for confidence score distributions

## Entity Resolution Process

The entity resolution process uses fuzzy string matching to identify potential matches across datasets:

1. Compare entity identifiers between datasets using string similarity metrics
2. Assign confidence scores to potential matches
3. Connect matched entities in the graph with confidence-weighted edges
4. Resolve conflicts using a priority-based system where incident data > telecom data > social media data

## Extensions and Future Work

- Implement more sophisticated entity resolution algorithms
- Add temporal analysis capabilities
- Incorporate text analytics for message content
- Apply machine learning for predictive analytics
- Create additional visualization options for larger networks

## License

[MIT License](LICENSE)
