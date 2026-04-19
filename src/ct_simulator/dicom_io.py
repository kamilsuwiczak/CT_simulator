import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import numpy as np
import io
import datetime

def save_dicom(image_array, patient_name, comment, date_str):
    image_16bit = (image_array * 65535).astype(np.uint16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian 

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    ds.PatientName = patient_name
    ds.PatientID = "123456"
    ds.ContentDate = datetime.datetime.now().strftime('%Y%m%d')
    ds.ContentTime = datetime.datetime.now().strftime('%H%M%S.%f')[:10]
    ds.StudyDate = date_str.replace("-", "")
    ds.Modality = "OT" 
    ds.ImageComments = comment

    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.InstanceNumber = "1"

    ds.Rows, ds.Columns = image_16bit.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0 
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    
    ds.PixelData = image_16bit.tobytes()

    dicom_io = io.BytesIO()
    pydicom.dcmwrite(dicom_io, ds, write_like_original=False)
    dicom_io.seek(0)
    
    return dicom_io


def load_dicom(file_path):
    ds = pydicom.dcmread(file_path)
    
    patient_name = str(ds.PatientName) if 'PatientName' in ds else "Unknown"
    study_date = ds.StudyDate if 'StudyDate' in ds else "Unknown"
    comment = ds.ImageComments if 'ImageComments' in ds else ""

    image_array = ds.pixel_array.astype(np.float64)

    if ds.BitsAllocated == 16:
        image_array = image_array / 65535.0
    else:
        max_val = np.max(image_array)
        if max_val > 0:
            image_array = image_array / max_val
            
    return image_array, patient_name, comment, study_date

