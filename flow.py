# flow.py
"""
Module to visualize the recruiting assistant’s workflow.
It imports the graph from agent.py and exposes a function to return its Mermaid diagram.
"""

from agent import graph

def get_mermaid_diagram() -> str:
    # Uses the built-in get_graph() method and draw_mermaid_png() if available.
    try:
        diagram = graph.get_graph().draw_mermaid()
    except Exception as e:
        diagram = f"Visualization error: {e}"
    return diagram

if __name__ == "__main__":
    print(get_mermaid_diagram())
