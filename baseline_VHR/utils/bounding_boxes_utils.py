from baseline_VHR.utils.ensemble import Rectangle

def rectangle_intersect(first_box: list, second_box: list):
        """
        Method checks two bounding boxes for intersection

        """
        a = Rectangle(*first_box)
        b = Rectangle(*second_box)
        area = a & b
        if area is None:
            return area, None
        else:
            intersect_area = area.calculate_area()
            # composite_bbox = a - area
            composite_bbox = a.difference(area)
            ratio_1, ratio_2 = intersect_area / a.calculate_area(), intersect_area / b.calculate_area()
            return (ratio_1, ratio_2), composite_bbox