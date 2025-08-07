import streamlit as st
import requests

st.title("🗑️🚮 Waste Management♻️")
img_byt=None

cam=st.camera_input("Capture an img")
if cam:
    img_byt=cam.getvalue()
    st.image(img_byt,caption="hii")

if img_byt:
    dic={"file":img_byt}
    res=requests.post("http://127.0.0.1:8000/predict",files=dic)
    response=res.json()
    st.write(response["predicted_type"])

