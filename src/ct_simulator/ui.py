import datetime

import numpy as np
import streamlit as st
from PIL import Image

from .dicom_io import save_dicom
from .reconstruction import calculate_rmse, simulate_tomograph


@st.cache_data(show_spinner=False)
def simulate_tomograph_cached(image_array, n_scans, n_detectors, fan_angle_deg, use_filter=False):
    return simulate_tomograph(image_array, n_scans, n_detectors, fan_angle_deg, use_filter, None)


def run_simulation(image_array, n_scans, n_detectors, fan_angle_deg, use_filter, progress_bar):
    if progress_bar is None:
        return simulate_tomograph_cached(image_array, n_scans, n_detectors, fan_angle_deg, use_filter)
    return simulate_tomograph(image_array, n_scans, n_detectors, fan_angle_deg, use_filter, progress_bar)


def main():
    st.set_page_config(layout="wide", page_title="Symulator Tomografu (Model Rownolegly)")
    st.title("Symulator Tomografu Komputerowego")
    st.markdown("Implementacja transformaty Radona z uzyciem modelu rownoleglego oraz algorytmu Bresenhama.")

    st.sidebar.header("Konfiguracja parametrow")

    n_detectors = st.sidebar.slider("Liczba detektorow (n)", 10, 720, 180, step=10)
    n_scans = st.sidebar.slider("Liczba skanow (iteracji)", 10, 720, 180, step=10)
    fan_angle_deg = st.sidebar.slider("Rozpietosc wachlarza (stopnie)", 45, 360, 180, step=45)
    use_filter = st.sidebar.checkbox("Uzyj filtrowania sinogramu (Splot / Ram-Lak)")

    uploaded_file = st.sidebar.file_uploader("Wgraj obraz wejsciowy", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        max_size = 200
        image.thumbnail((max_size, max_size))
        image_array = np.array(image) / 255.0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image_array, caption="Obraz oryginalny", use_container_width=True, clamp=True)

        if st.sidebar.button("Uruchom symulacje", type="primary"):
            progress_bar = st.progress(0)
            sinogram, reconstructed = run_simulation(
                image_array, n_scans, n_detectors, fan_angle_deg, use_filter, progress_bar
            )
            progress_bar.empty()

            rmse_value = calculate_rmse(image_array, reconstructed)

            with col2:
                sinogram_display = np.interp(sinogram, (sinogram.min(), sinogram.max()), (0, 1))
                st.image(sinogram_display, caption="Sinogram", use_container_width=True, clamp=True)

            with col3:
                st.image(reconstructed, caption="Obraz wyjsciowy (Rekonstrukcja)", use_container_width=True, clamp=True)
                st.success(f"Blad sredniokwadratowy (RMSE): {rmse_value:.4f}")

            st.session_state["reconstructed"] = reconstructed

    if "reconstructed" in st.session_state:
        st.divider()
        st.subheader("Zapisz wynik jako DICOM")
        col1, col2, col3 = st.columns(3)

        with col1:
            patient_name = st.text_input("Imie i nazwisko pacjenta", "Jan Kowalski")
        with col2:
            study_date = st.date_input("Data badania", datetime.date.today())
        with col3:
            comments = st.text_input("Komentarz", "Badanie symulowane")

        dicom_buffer = save_dicom(st.session_state["reconstructed"], patient_name, comments, str(study_date))
        st.download_button(
            label="Pobierz plik .dcm",
            data=dicom_buffer,
            file_name="rekonstrukcja.dcm",
            mime="application/dicom",
        )

