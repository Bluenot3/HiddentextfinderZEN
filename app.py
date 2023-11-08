import cv2
import pytesseract
from pytesseract import Output

def check_color_similarity(color1, color2, threshold=10):
    return all(abs(c1 - c2) < threshold for c1, c2 in zip(color1, color2))

def find_hidden_text(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use pytesseract to get the bounding boxes of text
    d = pytesseract.image_to_data(gray, output_type=Output.DICT)
    n_boxes = len(d['level'])

    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])

        # Extract the text region
        roi = image[y:y+h, x:x+w]

        # Get the average color of the text region and the surrounding area
        text_color = roi.mean(axis=0).mean(axis=0)
        bg_roi = image[max(0, y-5):min(image.shape[0], y+h+5), max(0, x-5):min(image.shape[1], x+w+5)]
        bg_color = bg_roi.mean(axis=0).mean(axis=0)

        # Check if the text color is similar to the background color
        if check_color_similarity(text_color, bg_color):
            print(f"Hidden text found at box {i}: {d['text'][i]}")

# Example usage:
find_hidden_text('resume.jpg')
