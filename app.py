import streamlit as st
import numpy as np
import math
from PIL import Image
from skimage.transform import resize
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid
import datetime
import io

# ==========================================
# 1. ALGORYTMY I OBLICZENIA BAZOWE
# ==========================================

def bresenham(x0, y0, x1, y1, width, height):
    """Implementacja algorytmu Bresenhama do znajdowania punktów na linii."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            if 0 <= x < width and 0 <= y < height:
                points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            if 0 <= x < width and 0 <= y < height:
                points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
            
    if 0 <= x < width and 0 <= y < height:
        points.append((x, y))
    return points

def get_positions(R, alpha, l_angle, n_detectors, center_x, center_y):
    """Wyznacza pozycje emiterów i detektorów dla modelu równoległego."""
    emitters = []
    detectors = []
    
    # Krok kątowy pomiędzy detektorami w rozpiętości wachlarza l_angle
    step = l_angle / (n_detectors - 1) if n_detectors > 1 else 0
    start_gamma = -l_angle / 2

    for i in range(n_detectors):
        gamma = start_gamma + i * step
        
        # Pozycja emitera
        ex = int(center_x + R * math.cos(alpha + math.pi - gamma))
        ey = int(center_y + R * math.sin(alpha + math.pi - gamma))
        emitters.append((ex, ey))
        
        # Pozycja detektora
        dx = int(center_x + R * math.cos(alpha + gamma))
        dy = int(center_y + R * math.sin(alpha + gamma))
        detectors.append((dx, dy))
        
    return emitters, detectors

def generate_filter(size):
    """Generuje filtr Ram-Lak w domenie przestrzennej."""
    filt = np.zeros(size)
    center = size // 2
    for i in range(size):
        if i == center:
            filt[i] = 1.0
        elif (i - center) % 2 == 0:
            filt[i] = 0.0
        else:
            filt[i] = -1.0 / (math.pi ** 2 * (i - center) ** 2)
    return filt

def apply_filter(sinogram):
    """Aplikuje filtr do sinogramu (splot)."""
    filtered = np.zeros_like(sinogram)
    num_detectors = sinogram.shape[1]
    
    # Generujemy filtr
    filt = generate_filter(num_detectors * 2 + 1)
    
    for i in range(sinogram.shape[0]):
        row = sinogram[i, :]
        
        # Wykonujemy pełny splot
        conv_full = np.convolve(row, filt, mode='full')
        
        # Obliczamy punkt startowy, aby wyciąć idealnie środek
        center_start = (len(conv_full) - len(row)) // 2
        
        # Wycinamy środkową część o rozmiarze oryginalnego wiersza
        filtered_row = conv_full[center_start : center_start + len(row)]
        
        # Teraz rozmiary się zgadzają (180 -> 180)
        filtered[i, :] = filtered_row
        
    return filtered

def calculate_rmse(image1, image2):
    """Oblicza błąd średniokwadratowy (RMSE)."""
    return np.sqrt(np.mean((image1 - image2) ** 2))

# ==========================================
# 2. TRANSFORMATA RADONA I REKONSTRUKCJA
# ==========================================

@st.cache_data(show_spinner=False)
def simulate_tomograph(image_array, n_scans, n_detectors, l_angle_deg, use_filter=False, _progress_bar=None):
    height, width = image_array.shape
    center_x, center_y = width // 2, height // 2
    R = math.sqrt(center_x**2 + center_y**2)
    l_angle = math.radians(l_angle_deg)
    
    sinogram = np.zeros((n_scans, n_detectors))
    
    # 1. Transformata Radona (Tworzenie sinogramu)
    for scan in range(n_scans):
        alpha = scan * (math.pi / n_scans) # Skanowanie w zakresie 180 stopni
        emitters, detectors = get_positions(R, alpha, l_angle, n_detectors, center_x, center_y)
        
        for i in range(n_detectors):
            x0, y0 = emitters[i]
            x1, y1 = detectors[i]
            points = bresenham(x0, y0, x1, y1, width, height)
            
            # Addytywne pochłanianie
            ray_sum = sum([image_array[y, x] for x, y in points])
            # Normalizacja przez długość promienia, by uniknąć prześwietleń
            sinogram[scan, i] = ray_sum / len(points) if points else 0
            
        if _progress_bar:
            _progress_bar.progress((scan + 1) / (2 * n_scans), text="Generowanie sinogramu...")

    # Zapis oryginalnego sinogramu do wyświetlenia
    display_sinogram = sinogram.copy()
    
    # Filtrowanie
    if use_filter:
        sinogram = apply_filter(sinogram)
        
    # Normalizacja sinogramu po filtrze (może mieć wartości ujemne)
    sinogram = np.interp(sinogram, (sinogram.min(), sinogram.max()), (0, 1))

    # 2. Odwrotna Transformata Radona (Rekonstrukcja)
    reconstructed = np.zeros((height, width))
    weight_matrix = np.zeros((height, width))
    
    for scan in range(n_scans):
        alpha = scan * (math.pi / n_scans)
        emitters, detectors = get_positions(R, alpha, l_angle, n_detectors, center_x, center_y)
        
        for i in range(n_detectors):
            x0, y0 = emitters[i]
            x1, y1 = detectors[i]
            points = bresenham(x0, y0, x1, y1, width, height)
            
            val = sinogram[scan, i]
            for x, y in points:
                reconstructed[y, x] += val
                weight_matrix[y, x] += 1
                
        if _progress_bar:
            _progress_bar.progress(0.5 + (scan + 1) / (2 * n_scans), text="Rekonstrukcja obrazu...")
            
    # Normalizacja zrekonstruowanego obrazu
    np.divide(reconstructed, weight_matrix, out=reconstructed, where=weight_matrix!=0)
    reconstructed = np.interp(reconstructed, (reconstructed.min(), reconstructed.max()), (0, 1))
    
    return display_sinogram, reconstructed

# ==========================================
# 3. OBSŁUGA DICOM
# ==========================================

def save_dicom(image_array, patient_name, comment, date_str):
    # Skalowanie do 16 bitów dla standardu DICOM
    image_16bit = (image_array * 65535).astype(np.uint16)
    
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2' # CT Image Storage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName = patient_name
    ds.StudyDate = date_str.replace("-", "")
    ds.ImageComments = comment
    
    ds.Rows, ds.Columns = image_16bit.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.PixelData = image_16bit.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    # Zapis do bufora w pamięci (zamiast na dysk), aby Streamlit mógł to pobrać
    dicom_io = io.BytesIO()
    pydicom.filewriter.dcmwrite(dicom_io, ds)
    dicom_io.seek(0)
    return dicom_io

# ==========================================
# 4. INTERFEJS STREAMLIT
# ==========================================

def main():
    st.set_page_config(layout="wide", page_title="Symulator Tomografu (Model Równoległy)")
    st.title("Symulator Tomografu Komputerowego")
    st.markdown("Implementacja transformaty Radona z użyciem modelu równoległego oraz algorytmu Bresenhama.")
    
    st.sidebar.header("Konfiguracja parametrów")
    
    # Parametry domyślne zgodne z wymaganiami na 5.0
    n_detectors = st.sidebar.slider("Liczba detektorów (n)", 10, 720, 180, step=10)
    n_scans = st.sidebar.slider("Liczba skanów (iteracji)", 10, 720, 180, step=10)
    l_angle_deg = st.sidebar.slider("Rozpiętość wachlarza (stopnie)", 45, 360, 180, step=45)
    use_filter = st.sidebar.checkbox("Użyj filtrowania sinogramu (Splot / Ram-Lak)")
    
    uploaded_file = st.sidebar.file_uploader("Wgraj obraz wejściowy", type=["png", "jpg", "jpeg", "bmp"])
    
    if uploaded_file is not None:
        # Przetwarzanie obrazu wejściowego (skala szarości, zmiana rozmiaru dla optymalizacji)
        image = Image.open(uploaded_file).convert('L')
        # Zmniejszamy obraz, jeśli jest za duży, aby symulacja w Pythonie nie trwała wieki
        max_size = 200
        image.thumbnail((max_size, max_size))
        image_array = np.array(image) / 255.0 # Normalizacja 0-1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image_array, caption="Obraz oryginalny", use_column_width=True, clamp=True)
            
        if st.sidebar.button("Uruchom symulację", type="primary"):
            progress_bar = st.progress(0)
            
            # Obliczenia
            sinogram, reconstructed = simulate_tomograph(
                image_array, n_scans, n_detectors, l_angle_deg, use_filter, progress_bar
            )
            
            progress_bar.empty()
            
            # Obliczanie RMSE
            rmse_val = calculate_rmse(image_array, reconstructed)
            
            with col2:
                # Normalizacja sinogramu wyłącznie na potrzeby wyświetlenia obrazka
                sino_disp = np.interp(sinogram, (sinogram.min(), sinogram.max()), (0, 1))
                st.image(sino_disp, caption="Sinogram", use_column_width=True, clamp=True)
            
            with col3:
                st.image(reconstructed, caption=f"Obraz wyjściowy (Rekonstrukcja)", use_column_width=True, clamp=True)
                st.success(f"Błąd średniokwadratowy (RMSE): {rmse_val:.4f}")
                
            # Zapis do sesji w celu umożliwienia eksportu DICOM bez przeliczania
            st.session_state['reconstructed'] = reconstructed

    # Sekcja DICOM (Wymaganie 4.0)
    if 'reconstructed' in st.session_state:
        st.divider()
        st.subheader("Zapisz wynik jako DICOM")
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            patient_name = st.text_input("Imię i nazwisko pacjenta", "Jan Kowalski")
        with d_col2:
            study_date = st.date_input("Data badania", datetime.date.today())
        with d_col3:
            comments = st.text_input("Komentarz", "Badanie symulowane")
            
        dicom_buffer = save_dicom(st.session_state['reconstructed'], patient_name, comments, str(study_date))
        st.download_button(
            label="Pobierz plik .dcm",
            data=dicom_buffer,
            file_name="rekonstrukcja.dcm",
            mime="application/dicom"
        )

if __name__ == "__main__":
    main()