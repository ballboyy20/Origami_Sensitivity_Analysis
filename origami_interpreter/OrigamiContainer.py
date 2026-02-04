"""
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

class OrigamiContainer:
    def __init__(self, origami_filepath=None, coords=None, panels=None):
        self._origami_filepath = None
        self._origami_coords_orig = None
        self._origami_panels_orig = None
        self._origami_coords = None
        self._origami_panels = None

        if origami_filepath is None and (coords is None or panels is None):
            raise ValueError("OrigamiContainer must be instantiated using either a filepath or a native python representation.")
        if origami_filepath is not None and (coords is not None or panels is not None):
            raise ValueError("OrigamiContainer cannot be instantiated using both a filepath and a native python representation.")
        if origami_filepath is not None:
            self._extract_file(origami_filepath)
        else:
            self._extract_pyrepr(coords, panels)

    def _extract_file(self, origami_filepath):
        """
        Read the filepath for a given origami representation file and interpret it as an OrigamiContainer object.
        
        :param origami_filepath: Complete filepath to the origami representation file. Supported file types are:
            .obj - Wavefront OBJ file
            .fold - Origami Simulator file
            .svg - Scalable Vector Graphics file
            .json - Custom JSON format to represent native python origami representation (outlined in extract_pyrepr docstring)

        :return: OrigamiContainer object
        """
        # TODO: #1 Priority
        raise NotImplementedError

    def _extract_pyrepr(self, coords, panels):
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
        
        :return: OrigamiContainer object
        """
        self._origami_coords_orig = coords
        self._origami_panels_orig = panels
        self._origami_coords = coords
        self._origami_panels = panels
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
        # TODO: #2 Priority
        raise NotImplementedError
        return
    
    def _interpret_svg(self):
        # TODO: #1 Priority
        raise NotImplementedError
        return
    
    def _interpret_json(self):
        # TODO: #3 Priority
        raise NotImplementedError
        return
    
