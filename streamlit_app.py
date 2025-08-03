import streamlit as st
import requests 

st.title("🐱🐶 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, width=250)
    bytes_data = uploaded_file.read()

    MODAL_FASTAPI_ENDPOINT_URL = "https://modal.com/apps/twinklehandaa/main/deployed/catdog-classifier/classify/" # Replace with YOUR actual URL

    with st.spinner("Classifying..."):
        try:
            headers = {'Content-Type': uploaded_file.type} 

            response = requests.post(
                MODAL_FASTAPI_ENDPOINT_URL,
                data=bytes_data, 
                headers=headers, 
                timeout=60 
            )

            if response.status_code == 200:
                prediction_data = response.json()
                if "prediction" in prediction_data:
                    st.success(f"Prediction: {prediction_data['prediction']}")
                else:
                    st.error("Unexpected response format from server. Response: " + str(prediction_data))
            else:
                st.error(f"Error from server: Status Code {response.status_code}. Response: {response.text}")

        except requests.exceptions.ConnectionError as e:
            st.error(f"Could not connect to the Modal endpoint. Please check the URL and your internet connection: {e}")
        except requests.exceptions.Timeout:
            st.error("The request timed out. The Modal function might be taking too long to respond.")
        except requests.exceptions.RequestException as e:
            st.error(f"An unexpected error occurred during the request: {e}")


