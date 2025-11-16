#!/usr/bin/env python3
"""
Knowledge Graph Visualization Script
Health Knowledge Recommender Project

Creates interactive visualizations of the JSON-LD knowledge graph.
Supports multiple visualization modes and filtering options.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict

try:
    from pyvis.network import Network
    import networkx as nx
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False
    print("Warning: pyvis not installed. Install with: pip install pyvis networkx")

# Color schemes for different node types
NODE_COLORS = {
    'FASTStage': '#FF6B6B',           # Red
    'ADLCapability': '#4ECDC4',        # Teal
    'IADLCapability': '#45B7D1',       # Blue
    'FASTCapabilityMapping': '#FFA07A', # Light coral
    'Document': '#95E1D3',             # Mint
    'SectionContent': '#F9ED69',       # Yellow
    'ParagraphContent': '#F38181',     # Pink
    'TipContent': '#AA96DA',           # Purple
    'Annotation': '#FCBAD3',           # Light pink
    'KnowledgeGraphMetadata': '#A8E6CF' # Light green
}

NODE_SHAPES = {
    'FASTStage': 'diamond',
    'ADLCapability': 'box',
    'IADLCapability': 'box',
    'FASTCapabilityMapping': 'ellipse',
    'Document': 'database',
    'SectionContent': 'text',
    'ParagraphContent': 'text',
    'TipContent': 'star',
    'Annotation': 'dot',
    'KnowledgeGraphMetadata': 'triangle'
}


class KnowledgeGraphVisualizer:
    """Visualizes JSON-LD knowledge graphs interactively."""

    def __init__(self, jsonld_file: str, output_dir: str = "visualizations"):
        """
        Initialize visualizer.

        Args:
            jsonld_file: Path to JSON-LD knowledge graph file
            output_dir: Output directory for visualizations
        """
        self.jsonld_file = Path(jsonld_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load knowledge graph
        with open(self.jsonld_file, 'r', encoding='utf-8') as f:
            self.kg = json.load(f)

        self.graph = self.kg.get('@graph', [])
        self.context = self.kg.get('@context', {})

        # Index nodes by type
        self.nodes_by_type = defaultdict(list)
        for node in self.graph:
            node_type = node.get('@type', 'Unknown')
            self.nodes_by_type[node_type].append(node)

        logging.info(f"Loaded knowledge graph with {len(self.graph)} nodes")
        for node_type, nodes in self.nodes_by_type.items():
            logging.info(f"  {node_type}: {len(nodes)}")

    def create_full_graph(self, output_file: str = "full_knowledge_graph.html"):
        """
        Create visualization of the complete knowledge graph.

        Args:
            output_file: Output HTML file name
        """
        if not HAS_PYVIS:
            print("Error: pyvis not installed. Install with: pip install pyvis networkx")
            return

        print("Creating full knowledge graph visualization...")

        # Create network
        net = Network(
            height='900px',
            width='100%',
            bgcolor='#222222',
            font_color='white',
            directed=True
        )

        # Configure physics
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {"iterations": 150}
            }
        }
        """)

        # Add nodes
        for node in self.graph:
            node_id = node.get('@id', 'unknown')
            node_type = node.get('@type', 'Unknown')

            # Create label
            label = self._create_node_label(node)
            title = self._create_node_title(node)

            # Get color and shape
            color = NODE_COLORS.get(node_type, '#CCCCCC')
            shape = NODE_SHAPES.get(node_type, 'dot')

            # Determine size based on type
            size = self._get_node_size(node_type)

            net.add_node(
                node_id,
                label=label,
                title=title,
                color=color,
                shape=shape,
                size=size
            )

        # Add edges
        edge_count = 0
        for node in self.graph:
            node_id = node.get('@id', 'unknown')

            # Find all relationships
            for key, value in node.items():
                if key.startswith('@'):
                    continue

                # Check if value is a reference (has @id)
                if isinstance(value, dict) and '@id' in value:
                    target_id = value['@id']
                    net.add_edge(node_id, target_id, title=key, arrows='to')
                    edge_count += 1

                # Check if value is a list of references
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and '@id' in item:
                            target_id = item['@id']
                            net.add_edge(node_id, target_id, title=key, arrows='to')
                            edge_count += 1

        print(f"Added {len(self.graph)} nodes and {edge_count} edges")

        # Save
        output_path = self.output_dir / output_file
        net.save_graph(str(output_path))
        print(f"✓ Saved to {output_path}")

    def create_fast_stages_view(self, output_file: str = "fast_stages_view.html"):
        """
        Create visualization focused on FAST stages and their relationships.

        Args:
            output_file: Output HTML file name
        """
        if not HAS_PYVIS:
            return

        print("Creating FAST stages visualization...")

        net = Network(
            height='900px',
            width='100%',
            bgcolor='#222222',
            font_color='white',
            directed=True
        )

        # Filter relevant nodes
        relevant_types = {'FASTStage', 'FASTCapabilityMapping', 'ADLCapability', 'IADLCapability'}

        for node in self.graph:
            node_type = node.get('@type', 'Unknown')
            if node_type not in relevant_types:
                continue

            node_id = node.get('@id', 'unknown')
            label = self._create_node_label(node)
            title = self._create_node_title(node)
            color = NODE_COLORS.get(node_type, '#CCCCCC')
            shape = NODE_SHAPES.get(node_type, 'dot')
            size = self._get_node_size(node_type) * 1.5

            net.add_node(node_id, label=label, title=title, color=color, shape=shape, size=size)

        # Add edges
        for node in self.graph:
            if node.get('@type') not in relevant_types:
                continue

            node_id = node.get('@id', 'unknown')

            for key, value in node.items():
                if key.startswith('@'):
                    continue

                if isinstance(value, dict) and '@id' in value:
                    target_id = value['@id']
                    if any(n.get('@id') == target_id and n.get('@type') in relevant_types for n in self.graph):
                        net.add_edge(node_id, target_id, title=key, arrows='to')

        output_path = self.output_dir / output_file
        net.save_graph(str(output_path))
        print(f"✓ Saved to {output_path}")

    def create_content_view(self, output_file: str = "content_view.html"):
        """
        Create visualization focused on content and annotations.

        Args:
            output_file: Output HTML file name
        """
        if not HAS_PYVIS:
            return

        print("Creating content visualization...")

        net = Network(
            height='900px',
            width='100%',
            bgcolor='#222222',
            font_color='white',
            directed=True
        )

        # Filter content-related nodes
        relevant_types = {
            'Document', 'SectionContent', 'ParagraphContent', 'TipContent',
            'Annotation', 'FASTStage'
        }

        for node in self.graph:
            node_type = node.get('@type', 'Unknown')
            if node_type not in relevant_types:
                continue

            node_id = node.get('@id', 'unknown')
            label = self._create_node_label(node)
            title = self._create_node_title(node)
            color = NODE_COLORS.get(node_type, '#CCCCCC')
            shape = NODE_SHAPES.get(node_type, 'dot')
            size = self._get_node_size(node_type) * 1.3

            net.add_node(node_id, label=label, title=title, color=color, shape=shape, size=size)

        # Add edges
        for node in self.graph:
            if node.get('@type') not in relevant_types:
                continue

            node_id = node.get('@id', 'unknown')

            for key, value in node.items():
                if key.startswith('@'):
                    continue

                if isinstance(value, dict) and '@id' in value:
                    target_id = value['@id']
                    if any(n.get('@id') == target_id and n.get('@type') in relevant_types for n in self.graph):
                        net.add_edge(node_id, target_id, title=key, arrows='to')

                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and '@id' in item:
                            target_id = item['@id']
                            if any(n.get('@id') == target_id and n.get('@type') in relevant_types for n in self.graph):
                                net.add_edge(node_id, target_id, title=key, arrows='to')

        output_path = self.output_dir / output_file
        net.save_graph(str(output_path))
        print(f"✓ Saved to {output_path}")

    def create_stage_specific_view(
        self,
        stage_code: str,
        output_file: Optional[str] = None
    ):
        """
        Create visualization for a specific FAST stage and related content.

        Args:
            stage_code: FAST stage code (e.g., "FAST-4")
            output_file: Output HTML file name
        """
        if not HAS_PYVIS:
            return

        if output_file is None:
            output_file = f"stage_{stage_code.lower()}_view.html"

        print(f"Creating visualization for {stage_code}...")

        net = Network(
            height='900px',
            width='100%',
            bgcolor='#222222',
            font_color='white',
            directed=True
        )

        # Find the FAST stage node
        stage_node_id = f"fast:{stage_code}"

        # Collect related nodes
        related_node_ids = {stage_node_id}

        # Find mappings for this stage
        for node in self.graph:
            if node.get('@type') == 'FASTCapabilityMapping':
                fast_stage = node.get('fastStage', {})
                if isinstance(fast_stage, dict) and fast_stage.get('@id') == stage_node_id:
                    related_node_ids.add(node.get('@id'))
                    # Add capability
                    capability = node.get('capability', {})
                    if isinstance(capability, dict):
                        related_node_ids.add(capability.get('@id'))

        # Find annotations mentioning this stage
        for node in self.graph:
            if node.get('@type') == 'Annotation':
                fast_stages = node.get('fastStages', [])
                for stage in fast_stages:
                    if isinstance(stage, dict) and stage.get('@id') == stage_node_id:
                        related_node_ids.add(node.get('@id'))
                        # Add annotated content
                        annotates = node.get('annotates', {})
                        if isinstance(annotates, dict):
                            related_node_ids.add(annotates.get('@id'))
                        break

        # Add nodes
        for node in self.graph:
            node_id = node.get('@id')
            if node_id not in related_node_ids:
                continue

            node_type = node.get('@type', 'Unknown')
            label = self._create_node_label(node)
            title = self._create_node_title(node)
            color = NODE_COLORS.get(node_type, '#CCCCCC')
            shape = NODE_SHAPES.get(node_type, 'dot')

            # Highlight the main stage
            size = self._get_node_size(node_type) * 2 if node_id == stage_node_id else self._get_node_size(node_type) * 1.5

            net.add_node(node_id, label=label, title=title, color=color, shape=shape, size=size)

        # Add edges between related nodes
        for node in self.graph:
            node_id = node.get('@id')
            if node_id not in related_node_ids:
                continue

            for key, value in node.items():
                if key.startswith('@'):
                    continue

                if isinstance(value, dict) and '@id' in value:
                    target_id = value['@id']
                    if target_id in related_node_ids:
                        net.add_edge(node_id, target_id, title=key, arrows='to')

                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and '@id' in item:
                            target_id = item['@id']
                            if target_id in related_node_ids:
                                net.add_edge(node_id, target_id, title=key, arrows='to')

        output_path = self.output_dir / output_file
        net.save_graph(str(output_path))
        print(f"✓ Saved to {output_path}")

    def create_statistics_view(self, output_file: str = "statistics.html"):
        """
        Create an HTML page with graph statistics and insights.

        Args:
            output_file: Output HTML file name
        """
        print("Creating statistics view...")

        # Calculate statistics
        stats = {
            'total_nodes': len(self.graph),
            'nodes_by_type': {k: len(v) for k, v in self.nodes_by_type.items()},
            'total_fast_stages': len(self.nodes_by_type.get('FASTStage', [])),
            'total_capabilities': len(self.nodes_by_type.get('ADLCapability', [])) +
                                len(self.nodes_by_type.get('IADLCapability', [])),
            'total_content': len(self.nodes_by_type.get('SectionContent', [])) +
                           len(self.nodes_by_type.get('ParagraphContent', [])) +
                           len(self.nodes_by_type.get('TipContent', [])),
            'total_annotations': len(self.nodes_by_type.get('Annotation', []))
        }

        # Create HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Graph Statistics</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            margin: 0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            padding: 40px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }}
        h1 {{
            text-align: center;
            font-size: 3em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            font-size: 1.2em;
            opacity: 0.9;
            margin-bottom: 40px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        .stat-card {{
            background: rgba(255, 255, 255, 0.2);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            transition: transform 0.3s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-number {{
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .breakdown {{
            margin-top: 40px;
        }}
        .breakdown-item {{
            background: rgba(255, 255, 255, 0.15);
            padding: 15px 25px;
            margin: 10px 0;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .breakdown-label {{
            font-size: 1.1em;
        }}
        .breakdown-value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        .color-indicator {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Knowledge Graph Statistics</h1>
        <div class="subtitle">Health Knowledge Recommender - PDF Extraction Analysis</div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Nodes</div>
                <div class="stat-number">{stats['total_nodes']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">FAST Stages</div>
                <div class="stat-number">{stats['total_fast_stages']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Capabilities</div>
                <div class="stat-number">{stats['total_capabilities']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Content Items</div>
                <div class="stat-number">{stats['total_content']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Annotations</div>
                <div class="stat-number">{stats['total_annotations']}</div>
            </div>
        </div>

        <div class="breakdown">
            <h2>Node Type Breakdown</h2>
        """

        for node_type, count in sorted(stats['nodes_by_type'].items(), key=lambda x: x[1], reverse=True):
            color = NODE_COLORS.get(node_type, '#CCCCCC')
            html += f"""
            <div class="breakdown-item">
                <div class="breakdown-label">
                    <span class="color-indicator" style="background-color: {color};"></span>
                    {node_type}
                </div>
                <div class="breakdown-value">{count}</div>
            </div>
            """

        html += """
        </div>
    </div>
</body>
</html>
        """

        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"✓ Saved to {output_path}")

    def _create_node_label(self, node: Dict) -> str:
        """Create a display label for a node."""
        node_type = node.get('@type', 'Unknown')

        if 'stageCode' in node:
            return node['stageCode']
        elif 'capabilityId' in node:
            return node['capabilityId']
        elif 'name' in node:
            return node['name'][:30]
        elif 'title' in node:
            return node['title'][:30]
        else:
            node_id = node.get('@id', 'unknown')
            return node_id.split(':')[-1][:20]

    def _create_node_title(self, node: Dict) -> str:
        """Create hover tooltip for a node."""
        lines = []
        node_type = node.get('@type', 'Unknown')
        lines.append(f"<b>Type:</b> {node_type}")

        # Add key properties
        for key in ['stageCode', 'stageName', 'capabilityId', 'name', 'title', 'impairmentLevel']:
            if key in node and node[key]:
                value = str(node[key])
                if len(value) > 100:
                    value = value[:100] + '...'
                lines.append(f"<b>{key}:</b> {value}")

        return '<br>'.join(lines)

    def _get_node_size(self, node_type: str) -> int:
        """Get size for a node based on its type."""
        sizes = {
            'FASTStage': 30,
            'ADLCapability': 25,
            'IADLCapability': 25,
            'FASTCapabilityMapping': 15,
            'Document': 35,
            'SectionContent': 20,
            'ParagraphContent': 15,
            'TipContent': 18,
            'Annotation': 12,
            'KnowledgeGraphMetadata': 40
        }
        return sizes.get(node_type, 15)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize JSON-LD Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all visualizations
  python visualize_knowledge_graph.py output/knowledge_graph.jsonld

  # Create specific visualization
  python visualize_knowledge_graph.py output/knowledge_graph.jsonld --view fast-stages

  # Create visualization for specific FAST stage
  python visualize_knowledge_graph.py output/knowledge_graph.jsonld --stage FAST-4

  # Custom output directory
  python visualize_knowledge_graph.py output/knowledge_graph.jsonld -o my_visualizations
        """
    )

    parser.add_argument(
        'jsonld_file',
        help='Path to JSON-LD knowledge graph file'
    )

    parser.add_argument(
        '-o', '--output',
        default='visualizations',
        help='Output directory for visualizations (default: visualizations)'
    )

    parser.add_argument(
        '--view',
        choices=['all', 'full', 'fast-stages', 'content', 'statistics'],
        default='all',
        help='Which visualization to create (default: all)'
    )

    parser.add_argument(
        '--stage',
        help='Create visualization for specific FAST stage (e.g., FAST-4)'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Check if file exists
    if not Path(args.jsonld_file).exists():
        print(f"Error: File not found: {args.jsonld_file}")
        sys.exit(1)

    # Create visualizer
    try:
        viz = KnowledgeGraphVisualizer(args.jsonld_file, args.output)
    except Exception as e:
        print(f"Error loading knowledge graph: {e}")
        sys.exit(1)

    print("\n" + "="*70)
    print("  Knowledge Graph Visualization")
    print("="*70 + "\n")

    # Create visualizations
    if args.stage:
        viz.create_stage_specific_view(args.stage)
    elif args.view == 'all':
        viz.create_full_graph()
        viz.create_fast_stages_view()
        viz.create_content_view()
        viz.create_statistics_view()
        print("\n" + "="*70)
        print("✓ All visualizations created successfully!")
        print(f"  Open files in: {viz.output_dir}")
        print("="*70)
    elif args.view == 'full':
        viz.create_full_graph()
    elif args.view == 'fast-stages':
        viz.create_fast_stages_view()
    elif args.view == 'content':
        viz.create_content_view()
    elif args.view == 'statistics':
        viz.create_statistics_view()


if __name__ == "__main__":
    main()
