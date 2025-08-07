# LiverAId - Karaciğer Hastalığı Risk Değerlendirme Sistemi

Siroz, HCC ve MAFLD risk değerlendirmesi için kapsamlı web tabanlı sistem.

## Dosyalar

```
liveraid-public/
├── app.py                    # Ana Flask uygulaması
├── database.py               # PostgreSQL veritabanı yönetimi
├── auth_utils.py             # Kimlik doğrulama
├── i18n.py                   # Dil yönetimi
├── medical_system_prompt.py  # AI asistan sistem promptları
├── requirements.txt          # Gerekli Python kütüphaneleri
├── pyproject.toml            # Proje yapılandırması
├── .env.example              # .env şablonu
├── .gitignore                # .gitignore
├── DOCUMENT_OCR_PROMPT       # OCR için AI prompt
├── PROMPT                    # Sonuçlar sayfasındaki AI değerlendirmesi için prompt
│
├── models/                   # Modeller
│   ├── __init__.py
│   ├── cirrhosis_model.py    # Siroz tahmin modeli
│   ├── hcc_model_final.py    # HCC tahmin modeli
│   ├── nafld_model.py        # MAFLD tahmin modeli
│   ├── *.pkl                 # Eğitilmiş model dosyaları
│   └── *.pkl                 # Scaler ve imputer dosyaları
│
├── static/                   # Web arayüzü
│   ├── css/
│   │   └── style.css         # Ana css dosyası
│   ├── js/
│   │   ├── main.js           # Ana JavaScript dosyası
│   │   ├── voice-input.js    # Ses girişi
│   │   ├── i18n.js           # Dil desteği
│   │   └── languages/        # Dil dosyaları
│   │       ├── tr.json       # Türkçe çeviriler
│   │       └── en.json       # İngilizce çeviriler
│   └── legal/                # Yasal belgeler
│       ├── privacy-tr.html   # Gizlilik politikası (TR)
│       ├── privacy-en.html   # Gizlilik politikası (EN)
│       ├── terms-tr.html     # Kullanım şartları (TR)
│       └── terms-en.html     # Kullanım şartları (EN)
│
├── templates/                # HTML şablonları
│   ├── base.html             # Ana şablon
│   ├── index.html            # Ana sayfa
│   ├── login.html            # Giriş sayfası
│   ├── register.html         # Kayıt sayfası
│   └── results.html          # Sonuçlar sayfası
│
└── data/                     # Veri setleri
    ├── dataset.csv
    ├── hcc_selected_final_converted.csv
    ├── hcc_selected_parameters.csv
    ├── guncelnafld_dataset_labeled.csv
    └── nafld_dataset_labeled.csv
```

## Sistem Gereksinimleri

- Python 3.9 veya üstü
- PostgreSQL 12 veya üstü
- Git

## 1. Projeyi Klonlama

```bash
git clone https://github.com/AhmetEsad/liveraid-public
cd liveraid-public
```

## 2. Python Virtual Env Oluşturma

```bash
# Sanal ortam oluştur
python3 -m venv venv

# Sanal ortamı aktifleştir (macOS/Linux)
source venv/bin/activate

# Sanal ortamı aktifleştir (Windows)
venv\Scripts\activate
```

## 3. Gerekli Kütüphaneleri Yükleme

```bash
pip install -r requirements.txt
```

## 4. PostgreSQL Veritabanı Kurulumu

### PostgreSQL Kurulumu (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### PostgreSQL Kurulumu (macOS)
```bash
brew install postgresql
brew services start postgresql
```

### PostgreSQL Kurulumu (Windows)
PostgreSQL'in resmi web sitesinden Windows installer'ı indirip kurun: https://www.postgresql.org/download/windows/

## 5. Veritabanı Kullanıcısı ve Veritabanı Oluşturma

PostgreSQL'e root kullanıcı olarak bağlanın:

```bash
sudo -u postgres psql
```

Yeni kullanıcı ve veritabanı oluşturun:

```sql
-- Yeni kullanıcı oluştur
CREATE USER envde_kullandiginiz_username WITH PASSWORD 'envde_kullandiginiz_sifre';

-- Veritabanı oluştur
CREATE DATABASE liver_assessment;

-- Kullanıcıya yetki ver
GRANT ALL PRIVILEGES ON DATABASE liver_assessment TO envde_kullandiginiz_username;

-- PostgreSQL'den çık
\q
```

## 6. .env dosyası

Proje klasöründe kendi değerlerinizle `.env` dosyası oluşturun:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=liver_assessment
DB_USER=envde_kullandiginiz_username
DB_PASSWORD=envde_kullandiginiz_sifre

GOOGLE_AI_API_KEY=gemini_api_key

FLASK_SECRET_KEY=secret_key
FLASK_ENV=production
FLASK_DEBUG=False
```

### Güvenlik Notları:
- `.env` dosyası asla Git'e commit edilmemeli (`.gitignore`'da zaten var)

## 7. API Key Alma

### Gemini
1. [Google AI Studio](https://aistudio.google.com/)'ya girin
2. Google hesabınızla giriş yapın
3. "Get API Key" butonuna tıklayın
4. Yeni bir API key oluşturun
5. API keyi `.env` dosyasındaki `GOOGLE_AI_API_KEY` değişkenine ekleyin

### OpenRouter
1. İnternette kaynaklar bulabilirsiniz. API key'i `.env`'de `OPENROUTER_API_KEY` olarak ekleyin.

### OpenAI
1. İnternette kaynaklar bulabilirsiniz. API key'i `.env`'de `OPENAI_API_KEY` olarak ekleyin.

## 8. Veritabanı Tablolarını Oluşturma

Uygulama ilk çalıştırıldığında gerekli tabloları otomatik olarak oluşturacaktır. Manuel olarak oluşturmak isterseniz:

```bash
python3 -c "from database import DatabaseManager; db = DatabaseManager()"
```

## 9. Uygulamayı Çalıştırma

```bash
python3 app.py
```
Uygulama http://localhost:5002 adresinde çalışacaktır.

## Veritabanı Yapısı

### Ana Tablolar:
- **users**: Kullanıcı bilgileri ve kimlik doğrulama
- **user_sessions**: Oturum yönetimi

Tablolar uygulama ilk çalıştırıldığında otomatik oluşturulur (`database.py`).

## Sorun Giderme

### 1. PostgreSQL Bağlantı Hatası
- PostgreSQL servisinin çalıştığını kontrol edin: `sudo systemctl status postgresql`
- `.env` dosyasındaki veritabanı bilgilerini kontrol edin
- PostgreSQL log dosyalarını kontrol edin: `/var/log/postgresql/`

### 2. Python Paket Hatası
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### 3. Google AI API Hatası
- API keyin doğru olduğunu kontrol edin
- Google AI Studio'da API kotanızı kontrol edin (ücretsiz limit dakikada 10 günde 500 istek olmalı)
- İnternet bağlantınızı kontrol edin

### 4. Model Dosyası Hatası
- `models/` klasöründe `.pkl` dosyalarının mevcut olduğunu kontrol edin
- Dosya izinlerini kontrol edin: `ls -la models/`

### 5. Static Dosya Hatası
- `static/` klasörü yapısının doğru olduğunu kontrol edin
- CSS/JS dosyalarının yüklendiğini browser devtoolstan kontrol edin
