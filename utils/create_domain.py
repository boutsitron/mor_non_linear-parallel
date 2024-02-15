"""Utilities for domain and mesh creation"""

import sys
import time

import gmsh
import numpy as np
from colorama import Fore
from firedrake import COMM_WORLD

import config

sys.path.append("..")

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank


def distance(x1: float, y1: float, x2: float, y2: float):
    """Computes the Euclidean distance between points (x1, y1) and (x2, y2)"""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def create_domain(
    inlet_positions: list[tuple],
    outlet_positions: list[tuple],
    buffer_length: float,
    lx: float,
    ly: float,
    meshfile: str,
    DIVISIONS: int,
    mesh_size: float = None,
    design_id: int = 3,
    non_design_id: int = 1,
):
    r"""
    Create domain and save to a .msh file for further use in Firedrake.

    Args:
        inlet_positions (list): positions of the inlet branches of the type (x1, y1, x2, y2) where
        (x1, y1) are the coordinates of the first node componing the inlet edge and
        (x2, y2)are the coordinates of the second node componing the edge. It is required
        that either x1 == x2 or y1 == y2 and that either x1 \in {0, lx} or x2 \in {0, ly},
        meaning that the passed coordinates should be points on the boundary of the
        design domain [0, lx] \times [0, ly]
        outlet_positions (list): positions of the outlet branches of the type (x1, y1, x2, y2) where
        (x1, y1) are the coordinates of the first node componing the inlet edge and
        (x2, y2) are the coordinates of the second node componing the edge. It is required
        that either x1 == x2 or y1 == y2 and that either x1 \in {0, lx} or x2 \in {0, ly},
        meaning that the passed coordinates should be points on the boundary of the
        design domain [0, lx] \times [0, ly]
        buffer_length (float): length of the inlet and outlet branches, meaning the lateral deviation from the
        coordinates passed to inlet_positions and outlet_positions
        lx (float): x-length of the design domain
        ly (float): y-length of the design domain
        DIVISIONS (int): number of divisions to use in the mesh
        meshfile (str): .msh file to be imported as a mesh, this is created before the start of the optimization
        by the method create_domain in utils/create_domain.py
        mesh_size (float): characteristic length of the mesh. Defaults to None, in which case it is set to
        min(lx, ly) / DIVISIONS
        design_id (int): ID to utilize for the design domain, namely the inner rectangle [0, lx] \times [0, ly],
        defaults to 3
        non_design_id (int): ID to utilize for the non design domain, defaults to 1

    Return:
        inlet (dict): a dictionary of the inlets with key given by the ID of the inlet to use in Firedrake
        and pair a tuple given by (sign of the outward pressure direction depending on the position
        of the inlet, component of the test function which is not cancelled out in the weak formulation)
        outlet (list): a list with the IDs of the outlets to use in Firedrake
        design_id (int): ID of the design domain (unchanged with respect to the input variable design_id)
        non_design_id (int): ID to utilize for the non design domain,
        defaults to 1 (unchanged with respect to the input variable non_design_id)
    """
    if rank == 0:
        mesh_time = time.time()

        gmsh.initialize(sys.argv)

        # Set the number of threads for parallel computing
        gmsh.option.setNumber("General.NumThreads", 1)

        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 8)

        gmsh.model.add("domain_model")

        if mesh_size is None:
            mesh_size = min(lx, ly) / DIVISIONS

        # Create main domain
        domain = gmsh.model.occ.addRectangle(0, 0, 0, lx, ly)

        inlets, outlets = create_inlets_outlets(
            inlet_positions, outlet_positions, buffer_length, lx, ly
        )

        # Fragment the domain and inlets/outlets
        gmsh.model.occ.fragment([(2, domain)], [(2, i) for i in inlets + outlets])

        # Synchronize model with OCC to finalize the changes
        gmsh.model.occ.synchronize()

        surfaces = gmsh.model.occ.getEntities(dim=2)
        fluid = [surface_tag for _, surface_tag in surfaces]
        non_design_domain = [
            surface_tag for _, surface_tag in surfaces if surface_tag != domain
        ]
        domain = [surface_tag for _, surface_tag in surfaces if surface_tag == domain]
        gmsh.model.addPhysicalGroup(
            2, non_design_domain, non_design_id, "non_design_domain"
        )
        gmsh.model.addPhysicalGroup(2, domain, design_id, "domain")
        gmsh.model.addPhysicalGroup(2, fluid, -1, "fluid")

        lines = gmsh.model.occ.getEntities(dim=1)

        inlet, outlet, non_walls = boundary_splitting(
            inlet_positions, outlet_positions, buffer_length, lx, ly
        )

        # assert that a correct number of lines was captured in the above checks, raise an error otherwise
        assert len(non_walls) == len(inlet_positions) + len(outlet_positions), (
            f"During the above checks {len(non_walls)} edges were captured as invalid walls"
            f" but {len(inlet_positions) + len(outlet_positions)} total branches were given in input."
        )

        # get walls as all lines which are not inlets nor outlets nor invalid walls
        walls = [
            line_tag
            for _, line_tag in lines
            if line_tag not in inlet
            and line_tag not in outlet
            and line_tag not in non_walls
        ]

        # create physical group for each inlet
        for key in inlet:
            gmsh.model.addPhysicalGroup(1, [key], key)

        # create unique "inlet" physical group
        gmsh.model.addPhysicalGroup(1, list(inlet.keys()), 1, "inlet")

        # create physical group for each outlet
        for key in outlet:
            gmsh.model.addPhysicalGroup(1, [key], key)

        # create unique "outlet", "walls" and "non_walls" physical groups, the latter mostly for checks
        gmsh.model.addPhysicalGroup(1, outlet, 2, "outlet")
        gmsh.model.addPhysicalGroup(1, walls, 3, "walls")
        gmsh.model.addPhysicalGroup(1, non_walls, 100, "non_walls")

        # Generate 2D mesh
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        gmsh.model.mesh.generate(2)

        # Save the model
        print(f"Saving mesh to {config.MESHDIR + meshfile}")
        gmsh.write(config.MESHDIR + meshfile)

        gmsh.finalize()

        print(
            f"{Fore.GREEN}Domain generation .............. {time.time()-mesh_time : 2.2f} s{Fore.RESET}"
        )
    else:
        inlet = None
        outlet = None

    inlet = COMM_WORLD.bcast(inlet, root=0)
    outlet = COMM_WORLD.bcast(outlet, root=0)

    return inlet, outlet, design_id, non_design_id


def create_inlets_outlets(
    inlet_positions: list,
    outlet_positions: list,
    buffer_length: float,
    lx: float,
    ly: float,
):
    r"""Creates inlets and outlets

    Args:
        inlet_positions (list): list with inlets coordinates
        outlet_positions (list): list with outlets coordinates
        buffer_length (float): branch length popping out of the design domain [0, lx] \times [0, ly]
        lx (float): x-length of the design domain
        ly (float): y-length of the design domain

    Returns:
        inlets (list): list of inlets rectangles
        outlets (list): list of outlets rectangles
    """
    inlets = []
    outlets = []

    # Create inlet(s)
    for inlet_position in inlet_positions:
        x1, y1, x2, y2 = inlet_position
        assert (
            0 <= x1 <= lx and 0 <= x2 <= lx and 0 <= y1 <= ly and 0 <= y2 <= ly
        ), f"Passed inlet {inlet_position} is not valid, given the dimensions of the chip"
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0.0:
            assert x1 == 0 or x1 == lx
            if x1 == 0.0:
                inlet = gmsh.model.occ.addRectangle(
                    -buffer_length, y1, 0, buffer_length, dy
                )
            else:
                inlet = gmsh.model.occ.addRectangle(lx, y1, 0, buffer_length, dy)
        elif dy == 0.0:
            assert y1 == 0 or y1 == ly
            if y1 == 0.0:
                inlet = gmsh.model.occ.addRectangle(
                    x1, -buffer_length, 0, dx, buffer_length
                )
            else:
                inlet = gmsh.model.occ.addRectangle(x1, ly, 0, dx, buffer_length)
        inlets.append(inlet)

    # Create outlet(s)
    for outlet_position in outlet_positions:
        x1, y1, x2, y2 = outlet_position
        assert (
            0 <= x1 <= lx and 0 <= x2 <= lx and 0 <= y1 <= ly and 0 <= y2 <= ly
        ), f"Passed outlet {outlet_position} is not valid, given the dimensions of the chip"
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0.0:
            assert x1 == 0 or x1 == lx
            if x1 == 0.0:
                outlet = gmsh.model.occ.addRectangle(
                    -buffer_length, y1, 0, buffer_length, dy
                )
            else:
                outlet = gmsh.model.occ.addRectangle(lx, y1, 0, buffer_length, dy)
        elif dy == 0.0:
            assert y1 == 0 or y1 == ly
            if y1 == 0.0:
                outlet = gmsh.model.occ.addRectangle(
                    x1, -buffer_length, 0, dx, buffer_length
                )
            else:
                outlet = gmsh.model.occ.addRectangle(x1, ly, 0, dx, buffer_length)
        outlets.append(outlet)

    return inlets, outlets


def boundary_splitting(
    inlet_positions: list,
    outlet_positions: list,
    buffer_length: float,
    lx: float,
    ly: float,
):
    r"""Boundary splitting between inlet, outlet, walls and non_walls (interfaces)

    Args:
        inlet_positions (list): list with inlets coordinates
        outlet_positions (list): list with outlets coordinates
        buffer_length (float): branch length popping out of the design domain [0, lx] \times [0, ly]
        lx (float): x-length of the design domain
        ly (float): y-length of the design domain

    Returns:
        inlet (dict): dictionary with keys the IDs of the inlet and values a tuple with:
        1. sign of the normal to the boundary (1 or -1)
        2. component of the test function which is not cancelled out by
        the previous sign in the computation (0 or 1)
        outlet (list): list of outlets IDs
        non_walls (list): list of interfaces between branches and design domain to be cancelled out from the walls for
        correct BCs imposition
    """
    lines = gmsh.model.occ.getEntities(dim=1)

    inlet = {}
    outlet = []
    non_walls = []
    # cycle over the lines
    for line_dim, line_tag in lines:
        update_inlet_dict(
            inlet, line_dim, line_tag, inlet_positions, buffer_length, lx, ly
        )
        update_outlet_dict(
            outlet, line_dim, line_tag, outlet_positions, buffer_length, lx, ly
        )

    get_non_walls(non_walls, lines, inlet_positions, outlet_positions)

    return inlet, outlet, non_walls


def update_inlet_dict(
    inlet: dict,
    line_dim: int,
    line_tag: int,
    inlet_positions: list,
    buffer_length: float,
    lx: float,
    ly: float,
):
    """Update inlet dictionary with direction of outward normal for BCs imposition and component of test function
       to be used for pressure drop

    Args:
        inlet (dict): inlet dictionary to be updated
        line_dim (int): always 1
        line_tag (int): tag of the current line under examination
        inlet_positions (list): list with all the inlet positions
        buffer_length (float): buffer length
        lx (float): x-length of the design domain
        ly (float): y-length of the design domain
    """
    # cycle over the inlets (compute sign and component of stress which needs to be set in BCs)
    # get center of mass and length of the edge
    com = gmsh.model.occ.getCenterOfMass(line_dim, line_tag)
    mass = gmsh.model.occ.getMass(line_dim, line_tag)

    for inlet_position in inlet_positions:
        x1, y1, x2, y2 = inlet_position
        if x1 == 0 and x2 == 0:  # this inlet is on the left edge
            component = 0
            sign = 1
            com_inlet = [-buffer_length, (y1 + y2) / 2.0]
        elif y1 == 0 and y2 == 0:  # this inlet is on the bottom edge
            component = 1
            sign = 1
            com_inlet = [(x1 + x2) / 2.0, -buffer_length]
        elif y1 == ly and y2 == ly:  # this inlet is on the top edge
            component = 1
            sign = -1
            com_inlet = [(x1 + x2) / 2.0, ly + buffer_length]
        elif x1 == lx and x2 == lx:  # this inlet is on the right edge
            component = 0
            sign = -1
            com_inlet = [lx + buffer_length, (y1 + y2) / 2.0]

        mass_inlet = distance(x1, y1, x2, y2)

        # add inlet to the inlets IDs
        if (
            np.isclose(com[0], com_inlet[0])
            and np.isclose(com[1], com_inlet[1])
            and np.isclose(mass, mass_inlet)
            and line_tag not in inlet
        ):
            inlet[line_tag] = (sign, component)


def update_outlet_dict(
    outlet: list,
    line_dim: int,
    line_tag: int,
    outlet_positions: list,
    buffer_length: float,
    lx: float,
    ly: float,
):
    """Update outlet list

    Args:
        outlet (list): list to be updated
        line_dim (int): always 1
        line_tag (int): tag of the current line under examination
        outlet_positions (list): list with all the outlet positions
        buffer_length (float): buffer length
        lx (float): x-length of the design domain
        ly (float): y-length of the design domain
    """
    # cycle over the outlets
    com = gmsh.model.occ.getCenterOfMass(line_dim, line_tag)
    mass = gmsh.model.occ.getMass(line_dim, line_tag)
    for outlet_position in outlet_positions:
        x1, y1, x2, y2 = outlet_position
        if x1 == 0 and x2 == 0:  # this outlet is on the left edge
            com_outlet = [-buffer_length, (y1 + y2) / 2.0]
        elif y1 == 0 and y2 == 0:  # this outlet is on the bottom edge
            com_outlet = [(x1 + x2) / 2.0, -buffer_length]
        elif y1 == ly and y2 == ly:  # this outlet is on the top edge
            com_outlet = [(x1 + x2) / 2.0, ly + buffer_length]
        elif x1 == lx and x2 == lx:  # this outlet is on the right edge
            com_outlet = [lx + buffer_length, (y1 + y2) / 2.0]

        mass_outlet = distance(x1, y1, x2, y2)

        # add outlet to the outlets IDs
        if (
            np.isclose(com[0], com_outlet[0])
            and np.isclose(com[1], com_outlet[1])
            and np.isclose(mass, mass_outlet)
            and line_tag not in outlet
        ):
            outlet.append(line_tag)


def get_non_walls(
    non_walls: list, lines: list, inlet_positions: list, outlet_positions: list
):
    """Get invalid walls (interfaces)

    Args:
        non_walls (list): list to be updated with invalid walls
        lines (list): list of all 1d lines
        inlet_positions (list): list with all the inlet positions
        outlet_positions (list): list with all the outlet positions
    """
    # cycle over the lines to catch the lines which are boundaries of the design domain but are not
    # boundaries of the actual domain because of inlets and outlets branches
    for line_dim, line_tag in lines:
        # get center of mass and length of the edge
        com = gmsh.model.occ.getCenterOfMass(line_dim, line_tag)
        mass = gmsh.model.occ.getMass(line_dim, line_tag)

        # get same quantities for the input data of inlets
        for inlet_position in inlet_positions:
            x1, y1, x2, y2 = inlet_position
            com_inlet = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            mass_inlet = distance(x1, y1, x2, y2)

            # add the current line to the non_walls if all the data coincides
            if (
                np.isclose(com[0], com_inlet[0])
                and np.isclose(com[1], com_inlet[1])
                and np.isclose(mass, mass_inlet)
                and line_tag not in non_walls
            ):
                non_walls.append(line_tag)

        # get same quantities for the input data of outlets
        for outlet_position in outlet_positions:
            x1, y1, x2, y2 = outlet_position
            com_outlet = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            mass_outlet = distance(x1, y1, x2, y2)

            # add the current line to the non_walls if all the data coincides
            if (
                np.isclose(com[0], com_outlet[0])
                and np.isclose(com[1], com_outlet[1])
                and np.isclose(mass, mass_outlet)
                and line_tag not in non_walls
            ):
                non_walls.append(line_tag)
