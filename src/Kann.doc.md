**Algorithm 1**: Kahn's Topological Sort

**Input**: A directed acyclic graph (DAG) \( G = (V, E) \)

**Output**: A list \( L \) containing the vertices of \( G \) in topologically sorted order, or an indication that \( G \) contains a cycle.

1. **Initialize**:

   - Let \( L \) be an empty list that will contain the sorted elements.
   - Let \( S \) be a set of all vertices in \( V \) with no incoming edges (i.e., vertices with in-degree 0).

2. **Process Vertices**:

   - While \( S \) is not empty:
     - Remove a vertex \( n \) from \( S \).
     - Append \( n \) to \( L \).
     - For each vertex \( m \) such that there is an edge \( (n, m) \) in \( E \):
       - Remove edge \( (n, m) \) from the graph.
       - If \( m \) has no other incoming edges (in-degree becomes 0), add \( m \) to \( S \).

3. **Check for Cycles**:
   - If \( E \) is not empty (i.e., the graph still contains edges), then \( G \) has at least one cycle.
     - Return an error indicating that the graph has a cycle.
   - Otherwise, return \( L \) as the topologically sorted order of the vertices.
