import io

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid


def save_dicom(image_array, patient_name, comment, date_str):
    image_16bit = (image_array * 65535).astype(np.uint16)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
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

    dicom_io = io.BytesIO()
    pydicom.filewriter.dcmwrite(dicom_io, ds)
    dicom_io.seek(0)
    return dicom_io

