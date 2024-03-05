import cv2
import pytesseract

# Load the image
image = cv2.imread("dsada.jpg")

# Convert the image to grayscale (optional)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform OCR using pytesseract
extracted_text = pytesseract.image_to_string(gray_image)

# Print the extracted text
print("text " + extracted_text)
