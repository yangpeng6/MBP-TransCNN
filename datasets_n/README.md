# Data Preparing

1. The magnetic bright point data was annotated by assigning a pixel value of 255 to magnetic bright points and a value of 0 to non-magnetic bright points.

2. The dataset was augmented by executing the code "elastic_transform.py" in the "tool" folder.

3. The code "canny_edge.py" in the "tool" folder was executed to detect edges from the annotated magnetic bright point data.

4. The pixel values indicating magnetic bright points were converted from 255 to 1.

5. The original image, magnetic bright point labels, and magnetic bright point edges were saved as an npz file by executing the code "picture_to_npz_normlize_.py" in the "tool" folder.
