"""
Guides to .svg files and relevant library:
    - https://en.wikipedia.org/wiki/SVG
    - https://pypi.org/project/svgelements/

Guides to .obj files:
 	- https://all3dp.com/2/obj-file-format-simply-explained/
	- https://en.wikipedia.org/wiki/Wavefront_.obj_file
	- https://paulbourke.net/dataformats/obj/

.stl files are not currently recommended because they inherently
are limited to triangulated meshes, which presumably confuses the
current hinge indicating process in SensitivityAnalysis.py
    - https://all3dp.com/1/stl-file-format-3d-printing/
    - https://en.wikipedia.org/wiki/STL_(file_format)
    - https://www.fabbers.com/tech/STL_Format

"""

from pathlib import Path

import svgelements as svg
import numpy as np

class OrigamiContainer:
    def __init__(self, origami_filepath=None, coords=None, panels=None, name=None, verbose=False):
        self._origami_filepath = None
        self._origami_coords_orig = None
        self._origami_panels_orig = None
        self._origami_coords = None
        self._origami_panels = None
        self._origami_name = None

        if origami_filepath is None and (coords is None or panels is None):
            raise ValueError("OrigamiContainer must be instantiated using either a filepath or a native python representation.")
        if origami_filepath is not None and (coords is not None or panels is not None):
            raise ValueError("OrigamiContainer cannot be instantiated using both a filepath and a native python representation.")
        if origami_filepath is not None:
            self._extract_file(origami_filepath, name, verbose)
        else:
            self._extract_pyrepr(coords, panels, name)

    def _extract_file(self, origami_filepath, name, verbose):
        """
        Read the filepath for a given origami representation file and interpret it as an OrigamiContainer object.
        
        :param origami_filepath: Complete filepath to the origami representation file. Supported file types are:
            .obj - Wavefront OBJ file
            .fold - Origami Simulator file
            .svg - Scalable Vector Graphics file
            .json - Custom JSON format to represent native python origami representation (outlined in extract_pyrepr docstring)
        :param name: Custom name to be saved for this origami container. By default, uses the file stem.
            
        :return: OrigamiContainer object
        """
        self._origami_filepath = origami_filepath
        filepath = Path(origami_filepath)
        if isinstance(name, str):
            self._origami_name = name
        else:
            self._origami_name = filepath.stem

        file_extension = filepath.suffix.lower()

        if file_extension == ".obj":
            self._interpret_obj()
        elif file_extension == ".fold":
            self._interpret_fold()
        elif file_extension == ".svg":
            self._interpret_svg(verbose)
        elif file_extension == ".json":
            self._interpret_json()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types are: .obj, .fold, .svg, .json")

    def _extract_pyrepr(self, coords, panels, name):
        """
        Use an existing origami representation, native to python
        and used by the SensitivityAnalysis code. Formatted as follows:
        coords = 
            [
                [p1x, p1y, p1z],
                [p2x, p2y, p2z],
                ...,
                [pNx, pNy, pNz],
            ]
        panels = 
            [
                [p1, p2, p3],
                [p2, p3, p4],
                ...,
                [p3, p4, pN]
            ]
        )

        :param coords: list of lists, where each sublist contains 3 numeric values representing the x, y and z coordinates of a point
        :param panels: list of lists, where each sublist contains integer values representing the indices of points that make up a panel
        :param name: Custom name to be saved for this origami container. Has a default value.
        
        :return: OrigamiContainer object
        """
        self._origami_coords_orig = coords
        self._origami_panels_orig = panels
        self._origami_coords = coords
        self._origami_panels = panels
        if isinstance(name, str):
            self._origami_name = name
        else:
            self._origami_name = "default_pattern_name"

        return self
    
    def visualize_origami(self):
        """
        Visualize the origami pattern in 3D
        
        :param self: 
        """
        # TODO: #2 Priority
        raise NotImplementedError
        return
    
    def normalize_size(self, method="area", metric=1.0):
        """
        Adjust the size of the origami pattern. Scales uniformly in all directions relative to the origin.
        
        :param self:
        :param method: Technique by which to normalize the origami pattern. Below are the options:
            "area": Scale pattern to have area equal to metric value
            "diam": Scale pattern to have circumscribed diameter equal to metric value
            "std": Scale pattern such that the standard deviation of its points is equivalent to metric value
        :param metric: numeric value used by "method" attribute
        """
        # TODO: #3 Priority
        raise NotImplementedError

    def normalize_pos(self, method="center", point=None):
        """
        Adjust the position of the origami pattern.
        
        :param self: 
        :param method: Technique by which to reposition the origami pattern. Below are the options:
            "center": Move the pattern such that the average of all its points lies on the origin
            "point": Move the pattern such that the average of all its points lies on the provided point
        :param point: None if method == "center", otherwise a datastructure, convertable to a numpy array,
            of 3 numeric values for x, y and z coordinates
        """
        # TODO: #3 Priority
        raise NotImplementedError
    
    def get_pyrepr(self):
        if self._origami_coords is None or self._origami_panels is None:
            raise AttributeError("This origami container is missing coordinate or panel information:"+
                                 f"\n\tCoordinates:\t{self._origami_coords}\n\tPanels:\t\t{self._origami_panels}")
        return self._origami_coords, self._origami_panels
    
    def get_original_pyrepr(self):
        if self._origami_coords_orig is None or self._origami_panels_orig is None:
            raise AttributeError("This origami container is missing its original coordinate or panel information:"+
                                 f"\n\tCoordinates:\t{self._origami_coords_orig}\n\tPanels:\t\t{self._origami_panels_orig}")
        return self._origami_coords_orig, self._origami_panels_orig

    
    def export_obj(self, directory, filename):
        # TODO: #2 Priority
        raise NotImplementedError
        return
    
    def export_fold(self, directory, filename):
        # TODO: #1 Priority
        raise NotImplementedError
        return

    def export_svg(self, directory, filename):
        # TODO: #2 Priority
        raise NotImplementedError
        return
    
    def export_json(self, directory, filename):
        # TODO: #3 Priority
        raise NotImplementedError
        return
    
    def _interpret_obj(self):
        # TODO: #1 Priority
        raise NotImplementedError
        return
    
    def _interpret_fold(self):
        # Switch .fold extension to .txt and it works to read as a plaintext file
        # TODO: #2 Priority
        raise NotImplementedError
        return
    
    def _interpret_svg(self, verbose):
        svg_document = svg.SVG.parse(self._origami_filepath)

        point_count = 0
        coord_list = []
        edge_list = []
        for element in svg_document.elements():
            # Path elements include some of the following
            #   Guide, Path
            print(type(element), element)
            if isinstance(element, svg.Path):
                # Path sub-elements are considered PathSegments and can be the following:
                #   Line, Arc, CubicBezier, QuadraticBezier, Move, Close
                for sub_element in element:
                    print(type(sub_element), sub_element)
                    if isinstance(sub_element, svg.Line):
                        # Each PathSegment object has a .point(pos) method, where pos=0 produces the start point and pos=1 produces the end point
                        point_start = sub_element.point(0)
                        point_end = sub_element.point(1)
                        start = (point_start.x, point_start.y)
                        end = (point_end.x, point_end.y)
                        coord_list.append(start)
                        coord_list.append(end)

                        edge_pair = (point_count, point_count + 1)
                        edge_list.append(edge_pair)
                        point_count += 2

        if verbose:
            print(f"\nExtracted {len(coord_list)} unique coordinates and {len(edge_list)} edges from SVG file.")
            print(coord_list)
            print(edge_list)

        coord_array = np.stack(coord_list)
        edge_array = np.stack(edge_list)

        dist_array = np.zeros((len(coord_array), len(coord_array)))
        for i in range(len(coord_array)):
            for j in range(len(coord_array)):
                dist_array[i, j] = np.linalg.norm(coord_array[i] - coord_array[j])
        close_tolerance = np.max(dist_array) / 10000
        close_array = np.isclose(dist_array, 0, atol=close_tolerance)

        if verbose:
            print(f"\nFound pairwise distances between all coordinates and identified {(np.sum(close_array) - len(coord_array))/2} coordinates that were within a tolerance of {close_tolerance}.")
            print(close_array)
            print(dist_array)

        keep_map = np.full(len(coord_array), True, dtype=np.bool_)
        for i, close_row in enumerate(close_array):
            for j, close_val in enumerate(close_row[i:]):
                if i != j+i and close_val:
                    if verbose:
                        print(f"\nCoordinates {i} and {j+i} are within the close tolerance of {close_tolerance} and will be merged.")
                        print(f"\tCoordinate {i}: {coord_array[i]}")
                        print(f"\tCoordinate {j+i}: {coord_array[j+i]}")
                    edge_array[edge_array == j+i] = i
                    keep_map[j+i] = False

        if verbose:
            print(f"\nMerged {np.sum(~keep_map)} coordinates that were within a tolerance of {close_tolerance} and updated panel point references accordingly.")
            print(edge_array)
            print(keep_map)

        coords = []
        for i, keep_point in enumerate(keep_map):
            if keep_point:
                coords.append(coord_array[i].tolist()+[0])
            else:
                edge_array[edge_array > i] -= 1
        edges = edge_array.tolist()

        if verbose:
            print(f"\nRemoved duplicate coordinates and updated panel point references accordingly. Final count of unique coordinates is {len(coords)}.")
            print(coords)
            print(edges)

        # TODO: Generate a graph representation of the points, linked by edges, and identify panels as cycles in the graph.
        panels = edges
        
        self._origami_coords_orig = coords
        self._origami_panels_orig = panels
        self._origami_coords = coords
        self._origami_panels = panels
    
    def _interpret_json(self):
        # TODO: #3 Priority
        raise NotImplementedError
        return
    
