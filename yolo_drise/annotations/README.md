# Annotated Data Generation

Go to https://www.makesense.ai/

Export formats:

- VOC XML Format (used by Faster-RCNN): a single XML file for each image, providing a detailed and structured representation of the data. 
    - The class name of the object.
    - A bounding box specified by the coordinates of its top-left (xmin, ymin) and bottom-right (xmax, ymax) corners in pixels.
    - Optional additional information, such as pose, truncated, and difficult flags.
    
- YOLO: .txt extension with format object-class, x_center, y_center, width, height
    - object-class is an integer representing the class of the object.
    - x_center and y_center are the normalized x and y coordinates of the center of the bounding box, relative to the dimensions of the image. These values are between 0 and 1.
    - width and height are the normalized width and height of the bounding box, also between 0 and 1.
