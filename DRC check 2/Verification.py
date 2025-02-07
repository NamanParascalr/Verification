#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import json
import gdspy
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from collections import deque
from rtree import index
import re
import xml.etree.ElementTree as ET


# In[2]:


def extract_drc_rules_from_xml(xml_file):
    """Extracts DRC rules from an XML file and returns them as a dictionary with layer numbers."""
    
    with open(xml_file, "r", encoding="utf-8") as file:
        xml_content = file.read()
    
    # Parse XML and extract text inside <text> tag
    root = ET.fromstring(xml_content)
    text_content = root.find(".//text").text if root.find(".//text") is not None else ""
    
    # Extract layer mappings from .lydrc file
    layer_mapping = extract_layer_mappings(xml_file)
    
    return extract_drc_rules(text_content, layer_mapping)


# In[3]:


def extract_layer_mappings(xml_file):
    """Extracts layer mappings from .lydrc file (e.g., 'm1' -> '19')."""
    layer_mapping = {}

    with open(xml_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("m") and "=" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    metal_name = parts[0].strip()
                    layer_info = parts[1].strip().replace("input(", "").replace(")", "")
                    layer_number = layer_info.split(",")[0]  # Extract only the first number
                    layer_mapping[metal_name] = layer_number  # Store mapping

    print("Layer Mapping:", layer_mapping)  # Debugging
    return layer_mapping


# In[4]:


def extract_drc_rules(text, layer_mapping):
    """Extracts DRC rules from text and returns a dictionary of constraints with layer numbers."""
    
    rules = {}

    # Regular expressions for different constraints
    width_pattern = re.compile(r'(m\d+)\.width\((\d+\.\w+)\)\.output\("(.+?)", "(.+?)"\)')
    spacing_pattern = re.compile(r'(m\d+)\.space\((\d+\.\w+).+?\)\.output\("(.+?)", "(.+?)"\)')
    area_pattern = re.compile(r'(m\d+)\.with_area\((\d+\.\.+?\d+)\)\.output\("(.+?)", "(.+?)"\)')
    corner_spacing_pattern = re.compile(r'(m\d+)\.space\((\d+\.\w+), euclidian\).+?\.output\("(.+?)", "(.+?)"\)')

    # Extract width rules
    for match in width_pattern.findall(text):
        metal_name = match[0]
        layer_number = layer_mapping.get(metal_name, metal_name)  # Convert to layer number
        rules.setdefault(layer_number, {})["Width"] = float(match[1].replace("nm", "")) / 1000

    # Extract spacing rules
    for match in spacing_pattern.findall(text):
        metal_name = match[0]
        layer_number = layer_mapping.get(metal_name, metal_name)
        rules.setdefault(layer_number, {})["Spacing"] = float(match[1].replace("nm", "")) / 1000

    # Extract area rules
    for match in area_pattern.findall(text):
        metal_name = match[0]
        layer_number = layer_mapping.get(metal_name, metal_name)
        rules.setdefault(layer_number, {})["Area"] = float(match[1].split("..")[-1])

    # Extract corner spacing rules
    for match in corner_spacing_pattern.findall(text):
        metal_name = match[0]
        layer_number = layer_mapping.get(metal_name, metal_name)
        rules.setdefault(layer_number, {})["Corner Spacing"] = float(match[1].replace("nm", "")) / 1000

    print("Extracted Rules:", rules)  # Debugging
    return rules


# In[5]:


def load_layout(file_path):
    """Loads a GDSII layout file and returns the layout object."""
    return gdspy.GdsLibrary(infile=file_path)


# In[6]:


def calculate_width(polygon):
    """Calculates the width of a polygon (assuming it's a rectangle)."""
    bounds = polygon.get_bounding_box()
    return bounds[1][0] - bounds[0][0] if bounds is not None else None


# In[7]:


def calculate_area(polygon):
    """Calculates the area of a polygon."""
    if not polygon.polygons or len(polygon.polygons) == 0:
        return 0
    shapely_polygon = ShapelyPolygon(polygon.polygons[0])
    return shapely_polygon.area


# In[8]:


def check_spacing(polygon1, polygon2, min_spacing):
    """Checks if two polygons meet the minimum spacing requirement."""
    shapely_polygon1 = ShapelyPolygon(polygon1.polygons[0])
    shapely_polygon2 = ShapelyPolygon(polygon2.polygons[0])

    # Compute minimum distance between polygons
    distance = shapely_polygon1.distance(shapely_polygon2)

    # Return True if spacing is valid, False otherwise
    return distance >= min_spacing


# In[9]:


def min_corner_distance(poly1, poly2):
    """Computes the minimum distance between the corners (vertices) of two polygons."""
    coords1 = list(poly1.exterior.coords)
    coords2 = list(poly2.exterior.coords)

    # Compute pairwise distance between all corners (vertices)
    return min(np.linalg.norm(np.array(p1) - np.array(p2)) for p1 in coords1 for p2 in coords2)


# In[32]:


def validate_layer(polygons, layer_rules, layer_number):
    violations = []
    
    min_width = layer_rules["min_width"]
    min_area = layer_rules["min_area"]
    min_spacing = layer_rules["min_spacing"]
    min_corner2corner_spacing = layer_rules["min_corner2corner_spacing"]

    print(f"\n=== Validating Layer {layer_number} with {len(polygons)} polygons ===\n")

    # **Width & Area Checks**
    for i, polygon in enumerate(polygons):
        width = calculate_width(polygon)
        area = calculate_area(polygon)

        print(f"Polygon {i} in Layer {layer_number}:")
        print(f"  width = {width}, type: {type(width)}")
        print(f"  area = {area}, type: {type(area)}")
        print(f"  min_width = {min_width}, min_area = {min_area}")

        if width is not None and width < min_width:
            violations.append(f"Layer {layer_number} Violation: Polygon {i} width {width:.6f} < {min_width}")
            print(f"  ❌ Violation: Width {width:.6f} < {min_width}")

        if area is not None and area < min_area:
            violations.append(f"Layer {layer_number} Violation: Polygon {i} area {area:.6f} < {min_area}")
            print(f"  ❌ Violation: Area {area:.6f} < {min_area}")

        print("-" * 50)

    # **Use R-tree for optimized spacing checks**
    rtree_idx = index.Index()
    poly_dict = {}  # Store indexed polygons

    # Insert polygons into R-tree
    for i, poly in enumerate(polygons):
        shapely_poly = ShapelyPolygon(poly.polygons[0])
        poly_dict[i] = shapely_poly
        minx, miny, maxx, maxy = shapely_poly.bounds
        rtree_idx.insert(i, (minx, miny, maxx, maxy))

    # Check only close neighbors using R-tree
    for i, poly1 in poly_dict.items():
        minx, miny, maxx, maxy = poly1.bounds

        # Query R-tree for nearby polygons
        nearby = list(rtree_idx.intersection((minx - min_spacing, miny - min_spacing, 
                                              maxx + min_spacing, maxy + min_spacing)))

        for j in nearby:
            if i >= j:  # Avoid duplicate checks
                continue

            poly2 = poly_dict[j]

            # **General Spacing Check**
            spacing_violation = poly1.buffer(min_spacing / 2).intersects(poly2.buffer(min_spacing / 2))
            min_corner_dist = min_corner_distance(poly1, poly2)

            print(f"Checking Polygon {i} vs Polygon {j} in Layer {layer_number}:")
            print(f"  min_spacing = {min_spacing}, spacing_violation = {spacing_violation}")
            print(f"  min_corner_spacing = {min_corner2corner_spacing}, corner_spacing_violation = {min_corner_dist < min_corner2corner_spacing}")

            if spacing_violation:
                violations.append(f"Layer {layer_number} Violation: Polygon {i} and {j} spacing < {min_spacing}")
                print(f"  ❌ Violation: Polygon {i} and {j} spacing < {min_spacing}")

            # **Corner-to-Corner Spacing Check**
            if min_corner_dist < min_corner2corner_spacing:
                violations.append(f"Layer {layer_number} Violation: Polygon {i} and {j} corner spacing {min_corner_dist:.6f} < {min_corner2corner_spacing}")
                print(f"  ❌ Violation: Corner spacing {min_corner_dist:.6f} < {min_corner2corner_spacing}")

            print("-" * 50)

    return violations


# In[33]:


def validate_layout(layout, constraints):
    """Validates an entire layout against given constraints and returns a list of violations."""
    violations = []

    for cell in layout.cells.values():
        print(cell)
        layer_polygons = {}

        for polygon in cell.polygons:
            layer = polygon.layers[0]
            layer_polygons.setdefault(layer, []).append(polygon)
        
        for layer, polygons in layer_polygons.items():
            layer_str = str(layer)  # Convert layer number to string
            layer_rules = constraints.get(layer_str, {})
            if not layer_rules:  # Debugging missing layers
                print(f"WARNING: No constraints found for layer {layer_str} (from GDS file)")

            violations.extend(validate_layer(polygons, layer_rules, layer_str))

    return violations


# In[34]:


def report_violations(violations):
    """Prints the list of DRC violations."""
    if violations:
        print("DRC Violations Found:")
        for violation in violations:
            print(violation)
    else:
        print("Valid Layout: No DRC Errors.")


# In[36]:


def main(lydrc_file, layout_file):
    """Main function to perform DRC validation."""
    layout = load_layout(layout_file)  # Load the GDSII layout
    constraints = extract_drc_rules_from_xml(lydrc_file)  
    violations = validate_layout(layout, constraints)  # Validate layout

    if violations:
        print("\n".join(violations))
    else:
        print("Layout is valid!")


if __name__ == "__main__":
    main("T1_M0_M1_M2_M3.lydrc", "T1_M0_M1.gds")
    print("End of file 1")


# In[ ]:




