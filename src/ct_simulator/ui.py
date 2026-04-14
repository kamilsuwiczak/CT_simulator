import datetime

import numpy as np
import streamlit as st
from PIL import Image

from .dicom_io import save_dicom, load_dicom
from .reconstruction import analyze_rmse_statistics, calculate_rmse, simulate_tomograph


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

    uploaded_file = st.sidebar.file_uploader("Wgraj obraz wejsciowy", type=["png", "jpg", "jpeg", "bmp", "dcm"])

    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".dcm"):
            image_array, patient_name, comment, study_date = load_dicom(uploaded_file)
            st.sidebar.info(f"DICOM Info:\n\nPacjent: {patient_name}\n\nData: {study_date}\n\nKomentarz: {comment}")

            # Skalowanie w dół dla optymalizacji, podobnie jak w przypadku innych obrazów
            image = Image.fromarray(np.uint8(image_array * 255.0))
            max_size = 200
            image.thumbnail((max_size, max_size))
            image_array = np.array(image) / 255.0
        else:
            image = Image.open(uploaded_file).convert("L")
            max_size = 200
            image.thumbnail((max_size, max_size))
            image_array = np.array(image) / 255.0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image_array, caption="Obraz oryginalny", width="stretch", clamp=True)

        if st.sidebar.button("Uruchom symulacje", type="primary"):
            progress_bar = st.progress(0)
            sinogram, reconstructed = run_simulation(
                image_array, n_scans, n_detectors, fan_angle_deg, use_filter, progress_bar
            )
            progress_bar.empty()

            rmse_value = calculate_rmse(image_array, reconstructed)

            with col2:
                sinogram_display = np.interp(sinogram, (sinogram.min(), sinogram.max()), (0, 1))
                st.image(sinogram_display, caption="Sinogram", width="stretch", clamp=True)

            with col3:
                st.image(reconstructed, caption="Obraz wyjsciowy (Rekonstrukcja)", width="stretch", clamp=True)
                st.success(f"Blad sredniokwadratowy (RMSE): {rmse_value:.4f}")

            st.session_state["reconstructed"] = reconstructed

        if st.sidebar.button("Wykonaj analize statystyczna RMSE"):
            with st.spinner("Trwa analiza statystyczna..."):
                stats = analyze_rmse_statistics(image_array, n_scans, n_detectors, fan_angle_deg)

            st.divider()
            st.subheader("Analiza statystyczna RMSE")

            st.markdown("**1) Zmiana RMSE w kolejnych iteracjach odwrotnej transformaty Radona**")
            st.line_chart(np.array(stats["rmse_per_iteration"]))
            st.caption(f"Liczba iteracji: {len(stats['rmse_per_iteration'])}")

            st.markdown("**2) Zmiana RMSE przy zwiekszaniu dokladnosci probkowania**")
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.caption("Wplyw liczby skanow")
                st.line_chart(np.array(stats["sampling"]["n_scans"]["rmse"]))
                st.write(
                    {
                        "Liczba skanow": stats["sampling"]["n_scans"]["values"],
                        "RMSE": [round(v, 5) for v in stats["sampling"]["n_scans"]["rmse"]],
                    }
                )

            with col_b:
                st.caption("Wplyw liczby detektorow")
                st.line_chart(np.array(stats["sampling"]["n_detectors"]["rmse"]))
                st.write(
                    {
                        "Liczba detektorow": stats["sampling"]["n_detectors"]["values"],
                        "RMSE": [round(v, 5) for v in stats["sampling"]["n_detectors"]["rmse"]],
                    }
                )

            with col_c:
                st.caption("Wplyw kata wachlarza")
                st.line_chart(np.array(stats["sampling"]["fan_angle_deg"]["rmse"]))
                st.write(
                    {
                        "Kat wachlarza": stats["sampling"]["fan_angle_deg"]["values"],
                        "RMSE": [round(v, 5) for v in stats["sampling"]["fan_angle_deg"]["rmse"]],
                    }
                )

            st.markdown("**3) Zmiana RMSE po wlaczeniu filtrowania**")
            col_nf, col_f = st.columns(2)
            with col_nf:
                st.metric("RMSE bez filtrowania", f"{stats['filter_comparison']['without_filter']:.5f}")
            with col_f:
                st.metric("RMSE z filtrowaniem", f"{stats['filter_comparison']['with_filter']:.5f}")

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

