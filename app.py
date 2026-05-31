import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# 1. Cấu hình trang web
st.set_page_config(page_title="Face Recognition", page_icon="👤")
st.title("👤 Ứng dụng Nhận Diện Khuôn Mặt")

# 2. Load model (Dùng cache để model không bị load lại mỗi lần bấm nút)
@st.cache_resource
def load_model():
    # SỬA TÊN FILE TẠI ĐÂY cho khớp với file .keras bạn tải về từ Colab
    return tf.keras.models.load_model('FaceID_MobileNetV2_Transfer.keras')

try:
    model = load_model()
    st.success("Đã tải mô hình thành công!")
except Exception as e:
    st.error(f"Lỗi tải mô hình: {e}. Vui lòng kiểm tra lại file .keras")

# 3. Khai báo thông số
# QUAN TRỌNG: Sửa danh sách này theo đúng thứ tự các thư mục lúc train
class_names = ['Nguoi_A', 'Nguoi_B', 'Nguoi_C'] 
IMG_SIZE = (128, 128)

# Hàm xử lý dự đoán
def predict_image(image):
    # Tiền xử lý ảnh giống hệt trên Colab
    img = image.convert('RGB').resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    
    # Dự đoán
    prob = model.predict(arr)[0]
    top3 = np.argsort(prob)[::-1][:3]
    
    # In kết quả
    st.markdown("### Kết quả dự đoán:")
    st.success(f"**Top 1: {class_names[top3[0]]}** - Độ tin cậy: {prob[top3[0]]*100:.1f}%")
    
    st.write("**Top 3 khả năng cao nhất:**")
    for i in top3:
        st.write(f"- {class_names[i]}: {prob[i]*100:.1f}%")

# 4. Tạo giao diện Up ảnh / Chụp ảnh
option = st.radio("Chọn phương thức đầu vào:", ("📸 Chụp từ Camera", "📂 Tải ảnh từ máy"))

if option == "📂 Tải ảnh từ máy":
    uploaded_file = st.file_uploader("Chọn ảnh khuôn mặt...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", width=300)
        if st.button("Nhận diện", use_container_width=True):
            with st.spinner("Đang phân tích..."):
                predict_image(image)

elif option == "📸 Chụp từ Camera":
    camera_image = st.camera_input("Chụp ảnh bằng Webcam của bạn")
    if camera_image is not None:
        image = Image.open(camera_image)
        if st.button("Nhận diện ảnh vừa chụp", use_container_width=True):
            with st.spinner("Đang phân tích..."):
                predict_image(image)
