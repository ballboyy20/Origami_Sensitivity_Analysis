––––––––––––––––––––––––––––––––––––––––––––
ORIGAMI KINEMATIC SENSITIVITY ANALYSIS
––––––––––––––––––––––––––––––––––––––––––––

This repository implements a geometry-driven sensitivity analysis framework for rigid-panel origami structures. The primary goal is to quantify how small nodal perturbations propagate into fold-angle changes across an origami pattern, independent of actuation strategy or material tuning.

The framework is intended for research use, particularly for comparing different origami crease-pattern topologies and identifying designs that are intrinsically robust to perturbations. This is especially relevant for precision applications such as deployable optical arrays, where small geometric errors can lead to unacceptable performance degradation.

––––––––––––––––––––––––––––––––––––––––––––
HIGH-LEVEL OVERVIEW
––––––––––––––––––––––––––––––––––––––––––––

At a high level, the program performs the following steps:
	1.	Builds a rigid origami model from node coordinates and panel connectivity
	2.	Enforces panel rigidity using internal bar constraints
	3.	Models fold behavior using hinge elements with dihedral-angle kinematics
	4.	Assembles a global stiffness matrix from bar and hinge contributions
	5.	Performs eigenvalue analysis to isolate mechanism modes
	6.	Computes hinge-level sensitivity for selected deformation modes
	7.	Visualizes relative hinge sensitivity across the pattern

The output answers questions such as:

Which origami patterns amplify small perturbations the most?
Which hinges dominate deformation?
How localized or distributed is folding motion?

––––––––––––––––––––––––––––––––––––––––––––
CORE MODELING ASSUMPTIONS
––––––––––––––––––––––––––––––––––––––––––––

Panels are perfectly rigid
Hinges are ideal rotational joints with finite stiffness
Bars enforce panel rigidity via a compatibility formulation
Sensitivity is evaluated in the linearized kinematic regime
Results are topology- and geometry-driven, not actuator-dependent

––––––––––––––––––––––––––––––––––––––––––––
REPOSITORY STRUCTURE
––––––––––––––––––––––––––––––––––––––––––––

main.py
Entry point for example usage and experiments

source/helper_classes.py
Core geometric and mechanical element definitions

source/sensitivity_model.py
System assembly, eigenvalue analysis, and sensitivity logic

––––––––––––––––––––––––––––––––––––––––––––
KEY CLASSES AND THEIR ROLES
––––––––––––––––––––––––––––––––––––––––––––

NODE

Represents a vertex in 3D space.

Each Node:
	•	Stores Cartesian coordinates
	•	Contributes 3 degrees of freedom (DOFs)
	•	Is shared between panels to ensure kinematic consistency

––––––––––––––––––––––––––––––––––––––––––––

PANEL

Represents a rigid face of the origami pattern.

Each Panel:
	•	Holds references to Node objects
	•	May have any number of sides
	•	Is enforced as rigid indirectly via internal bar elements

––––––––––––––––––––––––––––––––––––––––––––

BARELEMENT

Enforces rigid-body behavior within a panel.

Each BarElement:
	•	Connects two nodes
	•	Produces one row of the compatibility matrix
	•	Penalizes in-plane stretching and shearing
	•	Uses high stiffness to approximate rigid panels

The compatibility formulation ensures rigid-body invariance and prevents spurious deformation modes.

––––––––––––––––––––––––––––––––––––––––––––

HINGEELEMENT

Models a fold between two panels sharing an edge.

Node convention:
	•	Nodes j and k define the hinge axis
	•	Nodes i and l are the out-of-plane wing nodes

Each HingeElement:
	•	Computes local hinge geometry
	•	Evaluates a signed dihedral angle
	•	Computes the Jacobian of the fold angle with respect to nodal motion

The hinge Jacobian answers the question:
“If I perturb a node in space, how much does this hinge’s fold angle change?”

––––––––––––––––––––––––––––––––––––––––––––
THE SENSITIVITYMODEL CLASS (MAIN INTERFACE)
––––––––––––––––––––––––––––––––––––––––––––

This is the primary class users interact with.

Initialization requires:
	•	A list of node coordinates
	•	A list of panel definitions using node indices

Nodes are created once and shared across panels to guarantee correct kinematics.

––––––––––––––––––––––––––––––––––––––––––––
GEOMETRY GENERATION
––––––––––––––––––––––––––––––––––––––––––––

The model automatically:
	•	Generates nodes from coordinates
	•	Assigns nodes to panels
	•	Creates internal bars to enforce panel rigidity
	•	Detects shared edges and generates hinge elements
	•	Rejects invalid topologies (e.g., edges shared by more than two panels)

––––––––––––––––––––––––––––––––––––––––––––
GLOBAL STIFFNESS ASSEMBLY
––––––––––––––––––––––––––––––––––––––––––––

The global stiffness matrix is assembled as:

K = Cᵀ K_b C + Jᵀ K_h J

Where:
	•	C is the bar compatibility matrix
	•	K_b is the diagonal bar stiffness matrix
	•	J is the hinge Jacobian matrix
	•	K_h is the diagonal hinge stiffness matrix

This formulation cleanly separates panel rigidity from fold kinematics.

––––––––––––––––––––––––––––––––––––––––––––
EIGENVALUE ANALYSIS
––––––––––––––––––––––––––––––––––––––––––––

The system solves a symmetric eigenvalue problem.

The first six modes correspond to rigid-body motion.
Subsequent low-energy modes correspond to internal mechanisms.

Eigenvectors represent nodal deformation patterns associated with each mode.

––––––––––––––––––––––––––––––––––––––––––––
SENSITIVITY ANALYSIS
––––––––––––––––––––––––––––––––––––––––––––

For a given mechanism mode v, hinge sensitivity is computed as:

s = J v

Each entry of s represents how strongly a given hinge participates in that mode.

The framework supports:
	•	Sensitivity of individual modes
	•	Combined sensitivity across multiple modes
	•	Normalization for cross-pattern comparison

This produces a hinge-level map of kinematic amplification.

––––––––––––––––––––––––––––––––––––––––––––
VISUALIZATION
––––––––––––––––––––––––––––––––––––––––––––

The plotting utilities:
	•	Render the origami pattern in 3D
	•	Color hinges from blue (low sensitivity) to red (high sensitivity)
	•	Scale hinge thickness with activity level
	•	Label only highly active hinges for clarity

This provides an intuitive visualization of sensitivity hot-spots.

––––––––––––––––––––––––––––––––––––––––––––
TYPICAL WORKFLOW
––––––––––––––––––––––––––––––––––––––––––––
	1.	Define node coordinates and panel connectivity
	2.	Construct the SensitivityModel
	3.	Assemble the global stiffness matrix
	4.	Solve for eigenvalues and eigenvectors
	5.	Select mechanism modes of interest
	6.	Compute hinge sensitivity
	7.	Visualize results

––––––––––––––––––––––––––––––––––––––––––––
RESEARCH APPLICATIONS
––––––––––––––––––––––––––––––––––––––––––––

This framework is well suited for:
	•	Comparing origami patterns based on robustness
	•	Identifying sensitivity hot-spots in large arrays
	•	Studying how topology affects kinematic amplification
	•	Screening candidate designs before high-fidelity simulation

Because sensitivity is geometry-driven, results generalize across:
	•	Actuation strategies
	•	Material choices (within rigid-panel assumptions)
	•	Uniform geometric scaling

––––––––––––––––––––––––––––––––––––––––––––
LIMITATIONS
––––––––––––––––––––––––––––––––––––––––––––

Linearized (small-displacement) analysis
No panel bending or face compliance
No contact or self-intersection handling
Idealized hinge behavior

These limitations are intentional and appropriate for comparative, topology-level analysis.

––––––––––––––––––––––––––––––––––––––––––––
AUTHOR
––––––––––––––––––––––––––––––––––––––––––––

Jake Sutton
January 2026

––––––––––––––––––––––––––––––––––––––––––––
