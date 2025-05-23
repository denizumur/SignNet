# Trafik İşareti Tanıma Uygulaması (Traffic Sign Classifier)

Bu uygulama, kullanıcıdan alınan trafik işareti görselini analiz ederek hangi trafik işareti olduğunu tahmin eder.  
Eğitimde GTSRB (German Traffic Sign Recognition Benchmark) veri seti kullanılmıştır ve model Keras ile eğitilmiştir.
dataset:https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data
---

##  Örnek

![Uygulama Ekran Görüntüsü]([img]https://i.imgur.com/uknY95K.png[/img])


---

##  Özellikler

-  CNN tabanlı trafik işareti sınıflandırma modeli
-  Görsel yükleyerek gerçek zamanlı tahmin
-  Tahmin sonucu ve güven skoru (%)
-  Streamlit tabanlı kullanıcı dostu arayüz

---

##  Model Bilgisi

- Model: 6 katmanlı ConvNet
- Girdi Boyutu: 50x50 px
- Çıkış: 43 trafik işareti sınıfı
- Format: `traffic_sign_model.keras`

> Eğitilmiş model Colab üzerinde oluşturulmuş ve bu projeye dahil edilmiştir.

---


