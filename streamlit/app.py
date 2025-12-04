"""
VRP Visualization Streamlit App

A web interface for:
1. Uploading VRP instance files
2. Visualizing the problem (depot, customers, lockers)
3. Running PACO/SA solvers
4. Displaying and plotting solution routes
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from vrp_parser import parse_instance, VRPInstance, get_node_type_name, get_node_color
from solver_runner import run_solver, AVAILABLE_SOLVERS, SIZE_OPTIONS, SolverResult


# Page configuration
st.set_page_config(
    page_title="VRP Solver",
    page_icon="🚚",
    layout="wide",
)

# Title and description
st.title("🚚 Vehicle Routing Problem Solver")
st.markdown("""
Upload a VRP instance file, visualize the problem, and solve it using optimization algorithms.

**Supported Solvers:**
- **PACO**: Parallel Ant Colony Optimization
- **SA**: Simulated Annealing
""")

# Sidebar for controls
st.sidebar.header("⚙️ Settings")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload VRP Instance File",
    type=["txt"],
    help="Upload a VRP-PL instance file in the standard format"
)

# Solver selection
solver = st.sidebar.selectbox(
    "Select Solver",
    options=AVAILABLE_SOLVERS,
    format_func=lambda x: {"paco": "PACO (Parallel ACO)", "sa": "SA (Simulated Annealing)"}.get(x, x),
)

# Size selection
size = st.sidebar.selectbox(
    "Problem Size (Parameter Set)",
    options=SIZE_OPTIONS,
    help="Select parameter preset based on problem size"
)

# Initialize session state
if 'instance' not in st.session_state:
    st.session_state.instance = None
if 'instance_content' not in st.session_state:
    st.session_state.instance_content = None
if 'solution' not in st.session_state:
    st.session_state.solution = None


def plot_instance(instance: VRPInstance, solution: SolverResult = None) -> plt.Figure:
    """
    Create a matplotlib figure showing the VRP instance and optionally the solution routes.
    
    Args:
        instance: The parsed VRP instance
        solution: Optional solver result with routes
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot routes if solution provided
    if solution and solution.success and solution.raw_routes:
        # Generate colors for routes
        route_colors = plt.cm.tab10(np.linspace(0, 1, len(solution.raw_routes)))
        
        all_nodes = instance.get_all_nodes()
        
        for route_idx, route in enumerate(solution.raw_routes):
            color = route_colors[route_idx]
            
            # Get coordinates for route nodes (raw_routes contains integer node IDs)
            route_x = []
            route_y = []
            for node_id in route:
                if isinstance(node_id, int) and 0 <= node_id < len(all_nodes):
                    node = all_nodes[node_id]
                    route_x.append(node.x)
                    route_y.append(node.y)
            
            # Plot route as lines
            if len(route_x) > 1:
                ax.plot(route_x, route_y, '-', color=color, linewidth=2, alpha=0.7,
                       label=f'Route {route_idx + 1}')
                
                # Add arrows to show direction
                for i in range(len(route_x) - 1):
                    dx = route_x[i+1] - route_x[i]
                    dy = route_y[i+1] - route_y[i]
                    ax.annotate('', 
                               xy=(route_x[i+1], route_y[i+1]),
                               xytext=(route_x[i], route_y[i]),
                               arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                               )
    
    # Plot depot (star marker)
    ax.scatter(instance.depot.x, instance.depot.y, 
              c=get_node_color(0), marker='*', s=400, zorder=5,
              edgecolors='black', linewidths=1.5, label='Depot')
    
    # Plot customers by type
    for ctype in [1, 2, 3]:
        customers_of_type = [c for c in instance.customers if c.node_type == ctype]
        if customers_of_type:
            xs = [c.x for c in customers_of_type]
            ys = [c.y for c in customers_of_type]
            ax.scatter(xs, ys, 
                      c=get_node_color(ctype), marker='o', s=100, zorder=4,
                      edgecolors='black', linewidths=0.5,
                      label=get_node_type_name(ctype))
            
            # Add customer labels
            for c in customers_of_type:
                ax.annotate(str(c.id), (c.x, c.y), 
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=8, alpha=0.7)
    
    # Plot lockers (square markers)
    if instance.lockers:
        locker_xs = [l.x for l in instance.lockers]
        locker_ys = [l.y for l in instance.lockers]
        ax.scatter(locker_xs, locker_ys,
                  c=get_node_color(4), marker='s', s=150, zorder=4,
                  edgecolors='black', linewidths=1,
                  label='Locker')
        
        # Add locker labels
        for l in instance.lockers:
            ax.annotate(f"L{l.id}", (l.x, l.y),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=8, alpha=0.7, fontweight='bold')
    
    # Styling
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('VRP Instance Visualization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    
    plt.tight_layout()
    return fig


def display_instance_info(instance: VRPInstance):
    """Display instance statistics in columns."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Customers", instance.num_customers)
    with col2:
        st.metric("Lockers", instance.num_lockers)
    with col3:
        st.metric("Vehicles", instance.num_vehicles)
    with col4:
        st.metric("Capacity", instance.vehicle_capacity)
    
    # Customer type breakdown
    st.subheader("Customer Types")
    type_counts = {}
    for c in instance.customers:
        type_name = get_node_type_name(c.node_type)
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    cols = st.columns(len(type_counts))
    for i, (type_name, count) in enumerate(type_counts.items()):
        with cols[i]:
            st.metric(type_name, count)


def display_solution_info(solution: SolverResult):
    """Display solution statistics."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Distance", f"{solution.objective:.2f}")
    with col2:
        st.metric("Vehicles Used", solution.vehicles)
    with col3:
        st.metric("Runtime", f"{solution.runtime:.2f}s")
    
    # Display routes
    st.subheader("📋 Routes")
    for i, route in enumerate(solution.routes):
        route_str = " → ".join(map(str, route))
        st.text(f"Route {i+1}: {route_str}")


# Main content area
if uploaded_file is not None:
    # Read and parse the file
    try:
        content = uploaded_file.read().decode('utf-8')
        instance = parse_instance(content)
        st.session_state.instance = instance
        st.session_state.instance_content = content
        st.session_state.solution = None  # Reset solution when new file uploaded
        
        st.success(f"✅ Loaded instance: {uploaded_file.name}")
        
    except Exception as e:
        st.error(f"❌ Error parsing file: {e}")
        st.session_state.instance = None
        st.session_state.instance_content = None

# Display instance if loaded
if st.session_state.instance is not None:
    instance = st.session_state.instance
    
    # Instance info
    st.header("📊 Instance Information")
    display_instance_info(instance)
    
    # Solve button
    st.header("🔧 Solve")
    
    if st.button("🚀 Run Solver", type="primary"):
        with st.spinner(f"Running {solver.upper()} solver..."):
            result = run_solver(
                st.session_state.instance_content,
                solver=solver,
                size=size,
            )
            st.session_state.solution = result
        
        if result.success:
            st.success("✅ Solution found!")
        else:
            st.error(f"❌ Solver failed: {result.error_message}")
    
    # Display solution if available
    if st.session_state.solution is not None and st.session_state.solution.success:
        st.header("📈 Solution")
        display_solution_info(st.session_state.solution)
    
    # Visualization
    st.header("🗺️ Visualization")
    
    fig = plot_instance(instance, st.session_state.solution)
    st.pyplot(fig)
    plt.close(fig)

else:
    # Show instructions when no file is loaded
    st.info("👆 Upload a VRP instance file using the sidebar to get started.")
    
    st.markdown("""
    ### Instance File Format
    
    The VRP-PL instance file should have the following structure:
    
    ```
    <num_customers> <num_lockers>
    <num_vehicles> <vehicle_capacity>
    <demand_1>
    <demand_2>
    ...
    <demand_n>
    <x> <y> <earliest> <latest> <service_time> <type>  # depot (type=0)
    <x> <y> <earliest> <latest> <service_time> <type>  # customer 1
    ...
    <x> <y> <earliest> <latest> <service_time> <type>  # locker 1 (type=4)
    <locker_assignment_row_1>
    <locker_assignment_row_2>
    ...
    ```
    
    **Customer Types:**
    - Type 1: Home delivery only
    - Type 2: Locker delivery only
    - Type 3: Flexible (home or locker)
    - Type 4: Locker node
    """)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**VRP-PL Solver** v1.0")
st.sidebar.markdown("Using PACO and SA algorithms")
