import os
import cv2
from PIL import Image
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from streamlit_image_comparison import image_comparison

model = YOLO('yolov8n.pt')

st.title("Image Segmentation with YOLOv8: A Web Integration for Object Detection")

def save_uploadedfile(uploadedfile):
    with open(os.path.join("./media-directory/", "selfie.jpg"), "wb") as f:
        f.write(uploadedfile.getbuffer())

def convert_to_jpg(uploaded_image):
    im = Image.open(uploaded_image)
    if im.mode in ("RGBA", "P"):
        im = im.convert("RGB")
    uploaded_image_path = os.path.join(parent_media_path, "uploaded_image.jpg")
    im.save(uploaded_image_path)

st.divider()

st.markdown('')
st.markdown('##### Segmented Pieces')

## Placeholder Image
parent_media_path = "media-directory"
img_file = 'bus.jpg'

## Application States
APPLICATION_MODE = st.sidebar.selectbox("Our Options",
                                        ["Take Picture", "Upload Picture"]
                                        )

## Selfie Image
if APPLICATION_MODE == "Take Picture":
    st.sidebar.write(
        """
            A computer aided application that segments your input image, built on 
            the powerful YOLOv8 object detection algorithm developed by *ultralytics*.

            Simply take a selfie and it gets segmentated in real time.
        """
    )
    picture = st.camera_input("Take a picture")
    st.markdown('')
    if picture:
        st.sidebar.divider()
        st.sidebar.image(picture, caption="Selfie")
        if st.button("Segment!"):
            ## Function to save image
            save_uploadedfile(picture)
            st.sidebar.success("Saved File")
            selfie_img = os.path.join(parent_media_path, "/selfie.jpg")
        st.write("Click on **Clear photo** to retake picture")
        img_file = './media-directory/selfie.jpg'
    st.divider()

elif APPLICATION_MODE == "Upload Picture":
    st.sidebar.write(
        """
            A computer aided application that segments your input image, built on 
            the powerful YOLOv8 object detection algorithm developed by *ultralytics*.

            Simply drop your image and it gets segmentated in real time.
        """
    )
    st.sidebar.divider()
    uploaded_file = st.sidebar.file_uploader("Drop a JPG/PNG file", accept_multiple_files=False, type=['jpg', 'png'])
    if uploaded_file is not None and uploaded_file.type != ".jpg":
        convert_to_jpg(uploaded_file)
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        new_file_name = "uploaded_image.jpg"
        with open(os.path.join(parent_media_path, new_file_name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        img_file = os.path.join(parent_media_path, new_file_name)
        st.sidebar.success("File saved successfully")
        print(f"File saved successfully to {os.path.abspath(os.path.join(parent_media_path, new_file_name))}")
    else:
        st.sidebar.write("You are using a placeholder image, Upload your Image (.jpg for now) to explore")

results = model(img_file)
img = cv2.imread(img_file)
names_list = []
for result in results:
    boxes = result.boxes.cpu().numpy()
    numCols = len(boxes)
    if numCols > 0:
        cols = st.columns(numCols)
    else:
        print(f"Number of Boxes found: {numCols}")
        st.warning("Unable to id Distinct items - Please retry with a clearer Image")
    for box in boxes:
        r = box.xyxy[0].astype(int)
        rect = cv2.rectangle(img, r[:2], r[2:], (255, 55, 255), 2)
        
    st.markdown('')
    st.markdown('##### Slider of Uploaded Image and Segments')
    image_comparison(
        img1=img_file,
        img2=img,
        label1="Actual Image",
        label2="Segmented Image",
        width=700,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )
    for i, box in enumerate(boxes):
        r = box.xyxy[0].astype(int)
        crop = img[r[1]:r[3], r[0]:r[2]]
        predicted_name = result.names[int(box.cls[0])]
        names_list.append(predicted_name)
        with cols[i]:
            st.write(str(predicted_name) + ".jpg")
            st.image(crop)

st.sidebar.divider()
st.sidebar.markdown('')
st.sidebar.markdown('#### Distribution of identified items')

# Boolean to resize the dataframe, stored as a session state variable
st.sidebar.checkbox("Use container width", value=False, key="use_container_width")
if len(names_list) > 0:
    df_x = pd.DataFrame(names_list)
    summary_table = df_x[0].value_counts().rename_axis('unique_values').reset_index(name='counts')
    st.sidebar.dataframe(summary_table, use_container_width=st.session_state.use_container_width)
else:
    st.sidebar.warning("Unable to id Distinct items - Please retry with a clearer Image")

st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.sidebar.divider()

electronic = ['cell phone', 'laptop', 'tv', 'mouse', 'keyboard', 'remote', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
glass = ['wine glass', 'cup', 'bowl', 'fork', 'knife', 'spoon', 'bowl', 'cup', 'fork', 'knife', 'spoon']
vegetable = ['broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'orange', 'apple', 'banana', 'sandwich', 'orange', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'orange', 'apple', 'banana', 'sandwich']
plastic = ['bottle', 'cup', 'bowl', 'fork', 'knife', 'spoon', 'bowl', 'cup', 'fork', 'knife', 'spoon']
inorganic = ['cell phone', 'laptop', 'tv', 'mouse', 'keyboard', 'remote', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

if(any(item in names_list for item in glass or plastic)):
    st.markdown('')
    st.markdown('##### Non-Biodegradable Waste')
    st.write("Items like Glass, Plastic, etc. are classified as Non-Biodegradable Waste. They should be disposed of properly in blue dustbin.")
if(any(item in names_list for item in vegetable)):
    st.markdown('')
    st.markdown('##### Biodegradable Waste')
    st.write("Items like Vegetables, Fruits, etc. are classified as Biodegradable Waste. They should be disposed of properly in green dustbin.")
if(any(item in names_list for item in electronic or inorganic)):
    st.markdown('')
    st.markdown('##### Electronic or Inorganic Waste')
    st.write("Items like Electronic Gadgets, Books, etc. are classified as Electronic or Inorganic Waste. They should be disposed of properly in grey dustbin.")

st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.sidebar.divider()
st.sidebar.info("Made by VyomDwivedi")
