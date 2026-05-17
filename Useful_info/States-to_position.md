State-to-Location Mapping in TEM
States are integer indices, not raw coordinates. The mapping works in two stages:

1. Environment Discretizes the 2D Space into a Grid
In discritized_objects.py:121-141, the environment creates a regular Cartesian grid:

resolution_w = state_density × room_width (columns)
resolution_d = state_density × room_depth (rows)
xy_combination stores the (x,y) center of every grid cell as a 3D array of shape (resolution_d, resolution_w, 2)
n_states = resolution_w × resolution_d
2. Continuous Position → State Index via Nearest Neighbor
In discritized_objects.py:318-338, pos_to_state(x, y) finds the nearest grid node by Euclidean distance (np.argmin over squared distances) and returns its flattened index.

3. What the Agent Actually Receives
Each observation is a 3-tuple [state_id, one_hot_object_vector, continuous_pos]. The agent (whittington_2020.py:179-238) extracts the integer state_id and uses it as a location token — a discrete node ID on an implicit graph, not a coordinate.

Concrete Example (default params, state_density=1)
For a 10×10 room:

100 states (indices 0–99)
State 0 ≈ position (−4.5, −4.5), state 99 ≈ (4.5, 4.5)
Grid spacing = 1 unit
Bottom Line
TEM never sees (x, y) coordinates directly. It sees discrete node IDs that correspond to grid cells. The spatial structure is implicit — TEM must infer it from the sequence of visited states and their sensory observations.

Agent start position in all cases: random

this should be fixed at [0 0] the centre

Reward location should be at 3,3 in all environments