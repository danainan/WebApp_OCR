import cv2
import numpy as np
import streamlit as st
import keyboard
from PIL import Image
from tesserocr import PyTessBaseAPI
import streamlit.components.v1 as components
from transformers import AutoTokenizer , AutoModelForTokenClassification
from pythainlp.tokenize import word_tokenize
import torch
import os
import time
import streamlit.components.v1 as components


st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: center;
        bottom: 8px;
        width: 100%;
        color: white;
        text-align: left;
    }
    </style>
    <div class="footer">
    <p>Created by ❤️ <a href="https://github.com/danainan" target="_blank">@danainan</a></p>
    </div>
    """, unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
                MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <style>
        .container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .card {
            width: 100%;
            max-width: 500px;
        }
    </style>

    <div class="container">
        
        <img src="https://user-images.githubusercontent.com/71175110/220308869-f596631e-cd64-4a05-acf5-ca3a59f22966.jpg" class="img-fluid" alt="Responsive image">
    </div>
    </div>


    """,
    height=100,
)

def take_img():
    if st.button('Take Image'):
        global frame

        cap = cv2.VideoCapture(1)
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        st.title('Webcam Crop')
        x, y, w, h = 100, 100, 450, 250

        w = st.slider('Width', 0, 640, 450)
        h = st.slider('Height', 0, 480, 250)

        stframe = st.empty()
        stop = False
        st.write('Press "Q" to crop')

        while stop == False:
           ret, frame = cap.read()
           cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
           stframe.image(frame, channels='BGR')
           if keyboard.is_pressed('q'):
               stop = True
               crop = frame[y:y+h, x:x+w]
               cv2.imwrite('cropped/crop.jpg', crop)
               break


        cap.release()
        cv2.destroyAllWindows()

def upload_img():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image.save('cropped/crop.jpg')




# if st.button ('Take Image', use_container_width=True):
#     take_img()
    
option_img = st.selectbox('Choose your image',('Take Image','Upload Image'))
if option_img == 'Take Image':
    take_img()
elif option_img == 'Upload Image':
    upload_img()


image_path = 'cropped/crop.jpg'

if os.path.exists(image_path):
    image = Image.open(image_path)

    def original_img():
        st.write("Original Image")
        original = np.array(image)
        Image.fromarray(original).save("cropped/result.jpg")
        images = [image,original]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def binary_img():
        st.write("Binary")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        # kernel = np.ones((3, 3), np.uint8)
        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        Image.fromarray(thres).save("cropped/result.jpg")
        images = [image,thres]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def dilation_img():
        st.write("Dilation")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 0, 0],
                           [0, 1, 0]],np.uint8)

        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        new_img = cv2.dilate(thres, kernel, iterations=1)

        Image.fromarray(new_img).save("cropped/result.jpg")
        images = [image,new_img]
        
        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def erosion_img():
        st.write("Erosion")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 0, 0],
                           [0, 1, 0]],np.uint8)

        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        new_img = cv2.erode(thres, kernel, iterations=1)

        Image.fromarray(new_img).save("cropped/result.jpg")
        images = [image,new_img]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def opening_img():
        st.write("Opening")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 0, 0],
                           [0, 1, 0]],np.uint8)
        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        new_img = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel, iterations=1)

        Image.fromarray(new_img).save("cropped/result.jpg")
        images = [image,new_img]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def closing_img():
        st.write("Closing")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 0, 0],
                           [0, 1, 0]],np.uint8)

        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        new_img = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel, iterations=1)

        Image.fromarray(new_img).save("cropped/result.jpg")
        images = [image,new_img]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)
        


    option = st.selectbox(
    'Select Pre-Processing',
    ('Original Image','Binary', 'Dilation', 'Erosion', 'Opening', 'Closing'))

    if option == 'Original Image':
        original_img()
    elif option == 'Binary':
        binary_img()
    elif option == 'Dilation':
        dilation_img()
    elif option == 'Erosion':
        erosion_img()
    elif option == 'Opening':
        opening_img()
    elif option == 'Closing':
        closing_img()

    
    

def ocr_core(img):
    with PyTessBaseAPI(path='C:/Users/User/anaconda3/share/tessdata_best-main',lang="tha+eng") as api:
        api.SetImageFile(img)
        text = api.GetUTF8Text()
        text_array = []
        text_array.append(text.replace("\n", " "))
        return text_array,text
    
def ner_core(text):
        with st.spinner('Loading NER Model...'):
            time.sleep(1)
            name = 'thainer-corpus-v2-base-model'
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForTokenClassification.from_pretrained(name)

            sentence = text

            cut=word_tokenize(sentence.replace(" ", "<_>"))
            inputs=tokenizer(cut,is_split_into_words=True,return_tensors="pt")

            ids = inputs["input_ids"]
            mask = inputs["attention_mask"]
            # forward pass
            outputs = model(ids, attention_mask=mask)
            logits = outputs[0]

            predictions = torch.argmax(logits, dim=2)
            predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

            def fix_span_error(words,ner):
                _ner = []
                _ner=ner
                _new_tag=[]
                for i,j in zip(words,_ner):
                    #print(i,j)
                    i=tokenizer.decode(i)
                    if i.isspace() and j.startswith("B-"):
                        j="O"
                    if i=='' or i=='<s>' or i=='</s>':
                        continue
                    if i=="<_>":
                        i=" "
                    _new_tag.append((i,j))
                return _new_tag

            ner_tag=fix_span_error(inputs['input_ids'][0],predicted_token_class)
            print(ner_tag)

            def merge_name(ner_tag):
                _new_tag=[]
                _phrase=''
                _tag=''
                for i,j in ner_tag:
                    if j.startswith("B-"):
                        _phrase+=i
                        _tag=j
                    elif j.startswith("I-"):
                        _phrase+=i
                        _tag=j
                    else:
                        if _phrase!='':
                            _new_tag.append((_phrase,_tag))
                        _phrase=''
                        _tag=''
                return _new_tag
  
            ner_tag=merge_name(ner_tag)
            print('NAME NER ====>',ner_tag)

            person = []
            for i in ner_tag:
                if i[1].endswith("PERSON"):
                    person.append(i[0])

            # print('ผู้ส่ง :',person[0])
            # print('ผู้รับ :',person[1]+person[2])
            if len(person) == 2:
                print('ผู้ส่ง :',person[0]),print('ผู้รับ :',person[1])
            elif len(person) == 3:
                print('ผู้ส่ง :',person[0]),print('ผู้รับ :',person[1]+person[2])

            return person[0],person[1]




# st.write(ocr_core(os.path.join("cropped/result.jpg")))
if st.button ('OCR' ,use_container_width=True):
    img = os.path.join("cropped/result.jpg")
    text_array,text = ocr_core(img) 
    st.write(text_array)
    person1,person2 = ner_core(text)
    if person1 != '' and person2 != '':
        st.write('ผู้ส่ง :',person1)
        st.write('ผู้รับ :',person2)
        st.balloons()



    







    

