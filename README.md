# SignNet 
Trafik İşareti Tanıma ve Sınıflandırma Sistemi (CNN Tabanlı)

## Proje Açıklaması
SignNet, Alman Trafik İşaretleri Benchmark (GTSRB) veri seti üzerinde eğitilmiş bir derin öğrenme modelidir. Bu proje, trafik işareti görüntülerini sınıflandırmak için Convolutional Neural Network (CNN) mimarisi kullanır.

##  Kullanılan Teknolojiler
- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Google Colab
- Kaggle API

##  Veri Seti
- **Alman Trafik İşaretleri Benchmark (GTSRB)**
- Eğitim Verisi: 31,367 görüntü
- Test Verisi: 7,842 görüntü
- Toplam: 39,209 görüntü

Veri Kaggle API kullanılarak doğrudan Google Colab ortamına çekilmektedir.

##  Model Mimarisi
- 2 x Conv2D (32 ve 64 filtre, ReLU aktivasyon)
- MaxPooling2D (2x2)
- Dropout (%25) – Overfitting'i önlemek için
- Flatten + Dense
- Softmax çıkış katmanı (43 sınıf)

## Eğitim Sonuçları
Modelin eğitim sürecinde:
- Accuracy (Doğruluk)
- Loss (Kayıp)  
grafikleri görselleştirilerek kullanıcıya sunulmuştur.

##  İnference (Tahmin)
Model, `traffic_sign_model.h5` olarak kaydedilir ve test görüntüleri üzerinde tahmin yapmak için aşağıdaki fonksiyon kullanılır:

```python
predict_image("ornek_resim.png")

Fonksiyon çıkarım için görüntü dosyasını girdi olarak alacak, ilgili sınıf etiketini (label) ve olasılık değerini geri dönecektir. 
# Çıktı: ('Sınıf 14', 97.3)
