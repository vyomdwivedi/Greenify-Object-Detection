import os
import cv2 #pip install opencv-python
from PIL import Image
import pandas as pd #pip install pandas
import streamlit as st #pip install streamlit
from ultralytics import YOLO #pip install ultralytics
from streamlit_image_comparison import image_comparison #pip install streamlit-image-comparison

model = YOLO('yolov8n.pt')

st.title("Greenify - Waste Segmentation App")

def save_uploadedfile(uploadedfile):
    with open(os.path.join("./media-directory/", "bus.jpg"), "wb") as f:
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

parent_media_path = "media-directory"
img_file = 'bus.jpg'

APPLICATION_MODE = st.sidebar.selectbox("Our Options",
                                        ["Take Picture", "Upload Picture"]
                                        )

if APPLICATION_MODE == "Take Picture":
    st.sidebar.write(
        """
        """
    )
    picture = st.camera_input("Take a picture")
    st.markdown('')
    if picture:
        st.sidebar.divider()
        st.sidebar.image(picture, caption="Selfie")
        if st.button("Segment!"):
            save_uploadedfile(picture)
            st.sidebar.success("Saved File")
            selfie_img = os.path.join(parent_media_path, "/bus.jpg")
        st.write("Click on **Clear photo** to retake picture")
        img_file = './media-directory/bus.jpg'
    st.divider()

elif APPLICATION_MODE == "Upload Picture":
    st.sidebar.write(
        """
        Upload a .jpg or .png file (preferably with a clear background)
        1. Click on **Browse files** to upload your image
        2. Wait for the image to be processed
        3. View the segmented items and their distribution in the sidebar
        4. Dispose of waste responsibly based on the classification provided
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


names_list = []
try:
    results = model(img_file)
    img = cv2.imread(img_file)
    if img is None:
        st.error("Failed to load image. Please check the file format and try again.")
    else:
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
except Exception as e:
    st.error(f"An error occurred during model inference or image processing: {e}")

st.sidebar.divider()
st.sidebar.markdown('')
st.sidebar.markdown('#### Distribution of identified items')

st.sidebar.checkbox("Use container width", value=False, key="use_container_width")
if len(names_list) > 0:
    df_x = pd.DataFrame(names_list)
    summary_table = df_x[0].value_counts().rename_axis('unique_values').reset_index(name='counts')
    st.sidebar.dataframe(summary_table, use_container_width=st.session_state.use_container_width)
else:
    st.sidebar.warning("Unable to id Distinct items - Please retry with a clearer Image")

st.markdown('')
st.sidebar.divider()

electronic = [
    'cell phone', 'laptop', 'tv', 'mouse', 'keyboard', 'remote', 'microwave', 
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'scissors', 
    'teddy bear', 'hair drier', 'toothbrush',
    'tablet', 'printer', 'camera', 'headphones', 'smartwatch', 'speaker', 
    'game console', 'drone', 'charger', 'projector'
]
vegetable = [
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'orange', 
    'apple', 'banana', 'sandwich',
    'cucumber', 'lettuce', 'potato', 'tomato', 'grapes', 'mango', 'watermelon', 
    'pineapple', 'strawberry', 'pear'
]
plastic = [
    'bottle', 'cup', 'bowl', 'fork', 'knife', 'spoon',
    'straw', 'plastic bag', 'container', 'pen', 'comb', 'toothbrush cover', 
    'food wrap', 'detergent bottle', 'plastic chair', 'plastic toy'
]
inorganic = [
    'cell phone', 'laptop', 'tv', 'mouse', 'keyboard', 'remote', 'microwave', 
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'scissors', 
    'teddy bear', 'hair drier', 'toothbrush',
    'glass bottle', 'aluminum can', 'ceramic plate', 'steel spoon', 'metal pipe', 
    'tiles', 'light bulb', 'wire', 'mirror', 'battery'
]

if(any(item in names_list for item in plastic)):
    st.markdown('')
    st.markdown('### Non-Biodegradable Waste')
    st.write("Items like Plastic bottles, spoons etc. are classified as Non-Biodegradable Waste. They should be disposed of properly in blue dustbin.")
    st.info("♻️ **Eco Tip:** Rinse plastic containers before recycling. Avoid single-use plastics when possible. Reuse or repurpose plastic items, and participate in local recycling programs to ensure proper processing.")
elif(any(item in names_list for item in vegetable)):
    st.markdown('')
    st.markdown('### Biodegradable Waste')
    st.write("Items like Vegetables, Fruits, etc. are classified as Biodegradable Waste. They should be disposed of properly in green dustbin.")
    st.info("🌱 **Eco Tip:** Compost kitchen scraps and food waste to create natural fertilizer. Avoid mixing biodegradable waste with plastics or metals. Use compost bins or community composting facilities if available.")
elif(any(item in names_list for item in electronic or inorganic)):
    st.markdown('')
    st.markdown('### Electronic or Inorganic Waste')
    st.write("Items like Electronic Gadgets, Books, etc. are classified as Electronic or Inorganic Waste. They should be disposed of properly in grey dustbin.")
    st.info("🔋 **Eco Tip:** Never throw electronics or batteries in regular trash. Take e-waste to certified collection centers or recycling drives. Donate working electronics, and recycle metals and glass at appropriate facilities.")

st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.sidebar.divider()
st.sidebar.info("Made by VyomDwivedi")

