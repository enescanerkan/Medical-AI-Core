"""
Module: preprocessing/cv_transforms.py
Description: Contains advanced classical computer vision algorithms.
             Designed to prepare and enhance medical images before feeding them to deep learning models.
Açıklama: İleri seviye klasik bilgisayarlı görü algoritmalarını içerir.
          Medikal görüntüleri derin öğrenme modellerine vermeden önce hazırlamak ve iyileştirmek için tasarlanmıştır.
"""

import cv2
import numpy as np
import pywt


class AdvancedImageProcessor:
    """
    A utility class for image preprocessing techniques.
    Adheres to the Single Responsibility Principle (SRP) by only handling image transformations.

    Görüntü ön işleme teknikleri için bir yardımcı sınıf.
    Sadece görüntü dönüşümlerini işleyerek Tek Sorumluluk Prensibi'ne (SRP) uyar.
    """

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Crucial for enhancing tissue contrast in scans like mammograms or CTs.

        Kontrast Sınırlı Uyarlanabilir Histogram Eşitleme (CLAHE) uygular.
        Mamografi veya CT gibi taramalarda doku kontrastını artırmak için kritik öneme sahiptir.

        Args:
            image (np.ndarray): Input grayscale image. / Girdi siyah-beyaz görüntü.
            clip_limit (float): Threshold for contrast limiting. / Kontrast sınırlama eşiği.
            tile_grid (tuple): Size of grid for histogram equalization. / Histogram eşitleme için ızgara boyutu.

        Returns:
            np.ndarray: Contrast-enhanced image. / Kontrastı artırılmış görüntü.
        """
        # Create CLAHE object / CLAHE nesnesi oluştur
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(image)

    @staticmethod
    def get_discrete_wavelet(image: np.ndarray, wavelet_type: str = 'haar') -> tuple:
        """
        Performs 2D Discrete Wavelet Transform (DWT) to separate image frequencies.
        Useful for isolating calcifications or structural anomalies.

        Görüntü frekanslarını ayırmak için 2 Boyutlu Ayrık Dalgacık Dönüşümü (DWT) gerçekleştirir.
        Kalsifikasyonları veya yapısal anomalileri izole etmek için faydalıdır.

        Args:
            image (np.ndarray): Input grayscale image. / Girdi siyah-beyaz görüntü.
            wavelet_type (str): Type of wavelet to use (e.g., 'haar', 'db1'). / Kullanılacak dalgacık türü.

        Returns:
            tuple: Approximation (LL) and detail coefficients (LH, HL, HH).
                   Yaklaşım (LL) ve detay katsayıları (LH, HL, HH).
        """
        # Apply DWT / DWT uygula
        coeffs2 = pywt.dwt2(image, wavelet_type)
        return coeffs2

    @staticmethod
    def find_contours(image: np.ndarray, threshold_val: int = 127) -> list:
        """
        Detects contours in a medical image using basic binary thresholding.
        Serves as a preliminary step for ROI (Region of Interest) extraction or segmentation.

        Temel ikili eşikleme (binary thresholding) kullanarak medikal görüntüdeki kontürleri tespit eder.
        İlgi Alanı (ROI) çıkarımı veya segmentasyon için bir ön adım görevi görür.

        Args:
            image (np.ndarray): Input grayscale image. / Girdi siyah-beyaz görüntü.
            threshold_val (int): Pixel intensity threshold. / Piksel yoğunluk eşiği.

        Returns:
            list: A list of detected contours. / Tespit edilen kontürlerin bir listesi.
        """
        # Apply binary thresholding / İkili eşikleme uygula
        _, thresh = cv2.threshold(image, threshold_val, 255, cv2.THRESH_BINARY)

        # Find contours based on the thresholded image / Eşiklenmiş görüntüye dayanarak kontürleri bul
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours