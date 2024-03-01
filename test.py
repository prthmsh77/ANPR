import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import easyocr
import streamlit as st

plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')

def detect_plate(img): 
    plate_img = img.copy()
    roi = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)
    for (x, y, w, h) in plate_rect:
        roi_ = roi[y:y+h, x:x+w, :]
        plate = roi[y:y+h, x:x+w, :]
        cv2.rectangle(plate_img, (x+2, y), (x+w-3, y+h-5), (51, 181, 155), 3)
        
    return plate_img, plate

def display(img_, title=''):
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    st.pyplot()

def main():
    st.title("License Plate Detection and Recognition")
    st.sidebar.title("Options")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    use_image = st.sidebar.checkbox("Use Image")

    if use_image:
        st.sidebar.write("Using image...")
        image_file = st.sidebar.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
        if image_file is not None:
            image = np.array(bytearray(image_file.read()), dtype=np.uint8)
            img = cv2.imdecode(image, cv2.IMREAD_COLOR)
            output_img, plate = detect_plate(img)
            display(output_img, 'Detected license plate in the input image')

            # Perform OCR
            reader = easyocr.Reader(['en'])
            result = reader.readtext(plate)
            if result:
                st.write("License Plate Text:", result[0][-2].upper())
        else:
            st.info("Please upload an image file.")

if __name__ == '__main__':
    main()
