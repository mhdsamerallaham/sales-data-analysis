# Satış Verisi Analizi Projesi
# Gerçek dünya verisiyle veri temizleme, analiz ve görselleştirme

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Matplotlib için Türkçe font ayarı
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8')

print("📊 SATIŞ VERİSİ ANALİZİ PROJESİ")
print("=" * 50)

# 1. VERİ OLUŞTURMA (Gerçekçi Satış Verisi)
def create_realistic_sales_data(n_records=5000):
    """Gerçekçi satış verisi oluşturur"""
    
    np.random.seed(42)  # Tekrarlanabilir sonuçlar için
    
    # Ürün kategorileri ve fiyat aralıkları
    categories = {
        'Elektronik': (50, 2000),
        'Giyim': (20, 300),
        'Ev & Yaşam': (15, 500),
        'Kitap': (10, 100),
        'Spor': (25, 800),
        'Kozmetik': (30, 200)
    }
    
    # Müşteri segmentleri
    customer_segments = ['Yeni', 'Düzenli', 'VIP', 'Kurumsal']
    
    # Şehirler
    cities = ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya', 'Adana', 'Konya']
    
    data = []
    
    for i in range(n_records):
        # Tarih (son 2 yıl)
        start_date = datetime.now() - timedelta(days=730)
        random_days = np.random.randint(0, 730)
        date = start_date + timedelta(days=random_days)
        
        # Kategori seç
        category = np.random.choice(list(categories.keys()))
        price_range = categories[category]
        
        # Fiyat (kategoriye göre)
        base_price = np.random.uniform(price_range[0], price_range[1])
        
        # Miktar (1-10 arası, düşük miktarlar daha olası)
        quantity = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                  p=[0.4, 0.2, 0.15, 0.1, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01])
        
        # Toplam tutar
        total_amount = base_price * quantity
        
        # İndirim (bazen)
        discount_rate = np.random.choice([0, 0.05, 0.10, 0.15, 0.20], 
                                       p=[0.6, 0.15, 0.15, 0.07, 0.03])
        discount_amount = total_amount * discount_rate
        final_amount = total_amount - discount_amount
        
        # Müşteri bilgileri
        customer_segment = np.random.choice(customer_segments, 
                                          p=[0.3, 0.4, 0.2, 0.1])
        city = np.random.choice(cities)
        
        # Satış kanalı
        channel = np.random.choice(['Online', 'Mağaza', 'Telefon'], 
                                 p=[0.5, 0.4, 0.1])
        
        # Veri kaydı
        record = {
            'Sipariş_ID': f'ORD-{i+1000:05d}',
            'Tarih': date.strftime('%Y-%m-%d'),
            'Kategori': category,
            'Birim_Fiyat': round(base_price, 2),
            'Miktar': quantity,
            'Toplam_Tutar': round(total_amount, 2),
            'İndirim_Oranı': discount_rate,
            'İndirim_Tutarı': round(discount_amount, 2),
            'Net_Tutar': round(final_amount, 2),
            'Müşteri_Segmenti': customer_segment,
            'Şehir': city,
            'Satış_Kanalı': channel
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

# Veri setini oluştur
print("1️⃣ Veri seti oluşturuluyor...")
df = create_realistic_sales_data(5000)
print(f"✅ {len(df)} satış kaydı oluşturuldu!")

# 2. VERİ KEŞFETME (Exploratory Data Analysis)
print("\n2️⃣ VERİ KEŞFETME")
print("-" * 30)

# Temel bilgiler
print("📋 Veri Seti Genel Bilgileri:")
print(f"• Toplam kayıt sayısı: {len(df)}")
print(f"• Sütun sayısı: {len(df.columns)}")
print(f"• Tarih aralığı: {df['Tarih'].min()} - {df['Tarih'].max()}")

# İlk 5 kayıt
print("\n📄 İlk 5 Kayıt:")
print(df.head())

# Veri tipleri
print("\n🔍 Veri Tipleri:")
print(df.dtypes)

# Eksik veri kontrolü
print("\n❌ Eksik Veri Kontrolü:")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0] if missing_data.sum() > 0 else "✅ Eksik veri yok!")

# 3. VERİ TEMİZLEME
print("\n3️⃣ VERİ TEMİZLEME")
print("-" * 25)

# Tarih sütununu datetime'a çevir
df['Tarih'] = pd.to_datetime(df['Tarih'])

# Ay ve yıl sütunları ekle
df['Yıl'] = df['Tarih'].dt.year
df['Ay'] = df['Tarih'].dt.month
df['Ay_Adı'] = df['Tarih'].dt.strftime('%B')
df['Gün'] = df['Tarih'].dt.day_name()

# Aykırı değer kontrolü (IQR yöntemi)
def detect_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

outliers = detect_outliers(df, 'Net_Tutar')
print(f"🚨 Aykırı değer sayısı: {len(outliers)}")

print("✅ Veri temizleme tamamlandı!")

# 4. TEMEİL İSTATİSTİKLER
print("\n4️⃣ TEMEL İSTATİSTİKLER")
print("-" * 30)

# Sayısal sütunlar için özet istatistikler
numeric_columns = ['Birim_Fiyat', 'Miktar', 'Toplam_Tutar', 'Net_Tutar']
print("📊 Sayısal Veriler Özeti:")
print(df[numeric_columns].describe().round(2))

# Kategorik veriler
print("\n📈 Kategori Dağılımları:")
print(f"• Ürün Kategorileri: {df['Kategori'].value_counts().to_dict()}")
print(f"• Müşteri Segmentleri: {df['Müşteri_Segmenti'].value_counts().to_dict()}")
print(f"• Satış Kanalları: {df['Satış_Kanalı'].value_counts().to_dict()}")

# 5. ANALİZ VE GÖRSELLEŞTIRME
print("\n5️⃣ ANALİZ VE GÖRSELLEŞTİRME")
print("-" * 35)

# Grafik boyutlarını ayarla
plt.figure(figsize=(20, 15))

# 1. Aylık Satış Trendi
plt.subplot(3, 3, 1)
monthly_sales = df.groupby(['Yıl', 'Ay'])['Net_Tutar'].sum().reset_index()
# Pandas için İngilizce sütun isimleri kullan
monthly_sales_temp = monthly_sales.copy()
monthly_sales_temp['year'] = monthly_sales_temp['Yıl']
monthly_sales_temp['month'] = monthly_sales_temp['Ay'] 
monthly_sales_temp['day'] = 1
monthly_sales['Tarih'] = pd.to_datetime(monthly_sales_temp[['year', 'month', 'day']])
plt.plot(monthly_sales['Tarih'], monthly_sales['Net_Tutar'], marker='o', linewidth=2)
plt.title('Aylık Satış Trendi', fontsize=12, fontweight='bold')
plt.xlabel('Tarih')
plt.ylabel('Net Satış (TL)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 2. Kategori Bazında Satışlar
plt.subplot(3, 3, 2)
category_sales = df.groupby('Kategori')['Net_Tutar'].sum().sort_values(ascending=True)
plt.barh(category_sales.index, category_sales.values, color='skyblue')
plt.title('Kategori Bazında Toplam Satışlar', fontsize=12, fontweight='bold')
plt.xlabel('Net Satış (TL)')

# 3. Müşteri Segmenti Analizi
plt.subplot(3, 3, 3)
segment_sales = df.groupby('Müşteri_Segmenti')['Net_Tutar'].sum()
plt.pie(segment_sales.values, labels=segment_sales.index, autopct='%1.1f%%', startangle=90)
plt.title('Müşteri Segmenti Dağılımı', fontsize=12, fontweight='bold')

# 4. Şehir Bazında Satışlar
plt.subplot(3, 3, 4)
city_sales = df.groupby('Şehir')['Net_Tutar'].sum().sort_values(ascending=False)
plt.bar(city_sales.index, city_sales.values, color='lightcoral')
plt.title('Şehir Bazında Satışlar', fontsize=12, fontweight='bold')
plt.xlabel('Şehir')
plt.ylabel('Net Satış (TL)')
plt.xticks(rotation=45)

# 5. Satış Kanalları
plt.subplot(3, 3, 5)
channel_sales = df.groupby('Satış_Kanalı')['Net_Tutar'].sum()
plt.bar(channel_sales.index, channel_sales.values, color='lightgreen')
plt.title('Satış Kanalları Performansı', fontsize=12, fontweight='bold')
plt.xlabel('Kanal')
plt.ylabel('Net Satış (TL)')

# 6. Ortalama Sipariş Değeri (AOV)
plt.subplot(3, 3, 6)
aov_by_segment = df.groupby('Müşteri_Segmenti')['Net_Tutar'].mean().sort_values(ascending=False)
plt.bar(aov_by_segment.index, aov_by_segment.values, color='orange')
plt.title('Segment Bazında Ortalama Sipariş Değeri', fontsize=12, fontweight='bold')
plt.xlabel('Segment')
plt.ylabel('Ortalama Sipariş (TL)')

# 7. Haftanın Günlerine Göre Satışlar
plt.subplot(3, 3, 7)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_sales = df.groupby('Gün')['Net_Tutar'].sum().reindex(day_order)
plt.plot(range(len(day_sales)), day_sales.values, marker='o', linewidth=2, color='purple')
plt.title('Haftanın Günlerine Göre Satışlar', fontsize=12, fontweight='bold')
plt.xlabel('Gün')
plt.ylabel('Net Satış (TL)')
plt.xticks(range(len(day_sales)), ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz'])

# 8. İndirim Oranı Dağılımı
plt.subplot(3, 3, 8)
discount_dist = df['İndirim_Oranı'].value_counts().sort_index()
plt.bar(discount_dist.index, discount_dist.values, color='gold')
plt.title('İndirim Oranı Dağılımı', fontsize=12, fontweight='bold')
plt.xlabel('İndirim Oranı')
plt.ylabel('Sipariş Sayısı')

# 9. Fiyat vs Miktar İlişkisi
plt.subplot(3, 3, 9)
plt.scatter(df['Birim_Fiyat'], df['Miktar'], alpha=0.5, color='red')
plt.title('Birim Fiyat vs Miktar İlişkisi', fontsize=12, fontweight='bold')
plt.xlabel('Birim Fiyat (TL)')
plt.ylabel('Miktar')

plt.tight_layout()
plt.show()

# 6. ÖNEMLI BULGULAR VE İÇGÖRÜLER
print("\n6️⃣ ÖNEMLI BULGULAR")
print("-" * 25)

# En çok satan kategori
best_category = df.groupby('Kategori')['Net_Tutar'].sum().idxmax()
best_category_amount = df.groupby('Kategori')['Net_Tutar'].sum().max()

# En karlı müşteri segmenti
best_segment = df.groupby('Müşteri_Segmenti')['Net_Tutar'].sum().idxmax()

# En iyi performans gösteren şehir
best_city = df.groupby('Şehir')['Net_Tutar'].sum().idxmax()

# Ortalama sipariş değeri
avg_order_value = df['Net_Tutar'].mean()

# En popüler satış kanalı
popular_channel = df['Satış_Kanalı'].mode()[0]

print(f"🏆 En çok satan kategori: {best_category} ({best_category_amount:,.2f} TL)")
print(f"💎 En karlı müşteri segmenti: {best_segment}")
print(f"🌟 En iyi performans gösteren şehir: {best_city}")
print(f"💰 Ortalama sipariş değeri: {avg_order_value:.2f} TL")
print(f"📱 En popüler satış kanalı: {popular_channel}")

# Mevsimsel analiz
print(f"\n📅 Mevsimsel Analiz:")
seasonal_sales = df.groupby('Ay')['Net_Tutar'].sum()
best_month = seasonal_sales.idxmax()
worst_month = seasonal_sales.idxmin()
print(f"• En iyi ay: {best_month}. ay ({seasonal_sales[best_month]:,.2f} TL)")
print(f"• En kötü ay: {worst_month}. ay ({seasonal_sales[worst_month]:,.2f} TL)")

# 7. SONUÇ VE ÖNERİLER
print("\n7️⃣ SONUÇ VE ÖNERİLER")
print("-" * 30)

print("""
💡 ANALİZ SONUÇLARI:

📈 GÜÇLÜ YÖNLER:
• Düzenli müşteri segmenti en büyük gelir kaynağı
• Online satış kanalı güçlü performans gösteriyor
• Belirli kategoriler sürekli yüksek satış yapıyor

⚠️ İYİLEŞTİRME ALANLARI:
• Düşük performanslı şehirlerde pazarlama faaliyetleri artırılabilir
• Telefon satış kanalı güçlendirilmeli
• Mevsimsel dalgalanmalar için stok yönetimi optimize edilmeli

🎯 STRATEJİK ÖNERİLER:
• VIP müşteri segmentini büyütmek için özel kampanyalar
• Düşük satış yapan kategorilerde ürün çeşitliliği artırılabilir
• Hafta sonları için özel promosyonlar düzenlenebilir
""")

print("\n✅ ANALİZ TAMAMLANDI!")
print("📊 Bu analiz portfolyonuzda 'Veri Analizi ve Görselleştirme' projesi olarak kullanılabilir.")
print("🐙 GitHub'a yüklemek için kodu .py dosyası olarak kaydedin!")