# AI4CH — AI for Crop Health

A Django-based web platform for crop disease **classification** and **object detection** on beans, cassava, maize, and rice. Supports image and video uploads, EXIF-based location extraction, SMS notifications (Beem Africa), and bilingual content (English / Swahili).

---

## Table of Contents

- [Technology Stack](#technology-stack)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Authentication & Users](#authentication--users)
- [News & Reports](#news--reports)
- [Settings Reference](#settings-reference)
- [Reusable Modules & Consistency](#reusable-modules--consistency)
- [Configuration & Environment](#configuration--environment)
- [Getting Started](#getting-started)
- [URL Structure](#url-structure)
- [Important Notes & Gaps](#important-notes--gaps)

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Django 4.2, Python 3.x |
| **API** | Django REST Framework, Simple JWT |
| **Database** | MySQL (dev/prod via `config`), SQLite optional |
| **Auth** | Custom `User` (email login), django-allauth (Google), JWT for API |
| **ML / Vision** | PyTorch, TensorFlow (Keras), Ultralytics YOLO, OpenCV |
| **Static / Media** | WhiteNoise (compressed), django-resized (images) |
| **Frontend** | Server-rendered Django templates, Bootstrap 4 (Crispy Forms), TinyMCE & CKEditor |
| **i18n** | Django i18n (English, Swahili), `locale/` |
| **Config** | python-dotenv, `config.base` / `config.dev` / `config.prod` |
| **Location data** | `mtaa` (Tanzania regions, districts, wards, streets) |

### Main Dependencies (`requirements.txt`)

- **Web:** `Django`, `djangorestframework`, `django-cors-headers`, `django-allauth`, `whitenoise`
- **Auth:** `djangorestframework-simplejwt`, `PyJWT`, `cryptography` (Fernet for phone encryption)
- **Forms / UI:** `django-crispy-forms`, `django-countries`, `django-ckeditor`, `django-tinymce`, `django-resized`
- **ML / Vision:** `torch`, `torchvision`, `opencv-python`, `pillow`, `numpy`, `tensorflow_hub` (pulls TensorFlow/Keras for `.h5` models)
- **YOLO:** `ultralytics` — **not in requirements.txt**; add manually if needed
- **DB:** `mysqlclient`, `PyMySQL`
- **Utils:** `requests`, `python-dotenv`

### Dependencies to add manually (used in code but not in `requirements.txt`)

- **`ultralytics`** — YOLO detection in beans, cassava, maize, rice
- **`exifread`** — EXIF parsing in `img_loc`
- **`geopy`** — reverse geocoding in `img_loc`
- **`mtaa`** — Tanzania administrative hierarchy (regions → districts → wards → streets) for registration and location APIs

---

## Architecture Overview

- **Monolithic Django project** with multiple **feature apps**.
- **Two entry points:**
  - **Web UI:** session-based auth, Django templates (`templates/backend/`, `templates/interfaces/`).
  - **REST API:** JWT via Simple JWT; detection endpoints accept `file` + `user_id` (currently `AllowAny`).
- **Crop apps** (beans, cassava, maize, rice) share the same pattern: **Data** (upload + optional geo), **PredictionResult** (classification), **DetectionResult** (object detection); web views + one API view per crop that runs YOLO or Keras and returns JSON.
- **Config:** `config/base.py`, `config/dev.py`, `config/prod.py` hold env-driven values; `ai4ch/settings.py` imports them and sets DB, INSTALLED_APPS, REST_FRAMEWORK, etc.
- **Static:** `static/` is the source; `staticfiles/` is `collectstatic` output; WhiteNoise serves compressed static files.

**High-level flow:** User uploads image/video → stored in **CropData** → **YOLO** (detection) or **Keras** (classification) runs → result in **DetectionResult** / **PredictionResult** → web redirect or API JSON. Optionally EXIF location is extracted via `img_loc` and SMS sent via Beem Africa.

---

## Project Structure

```
ai4ch/
├── ai4ch/                 # Project package
│   ├── settings.py        # Main settings (uses config.base, config.dev/prod)
│   ├── urls.py            # Root URLconf: detection APIs outside i18n; rest under i18n_patterns
│   ├── asgi.py / wsgi.py
│   └── ...
├── config/
│   ├── base.py            # SECRET_KEY, mail, Beem SMS, FARNET_ENCRYPTION_KEY
│   ├── dev.py             # Dev DB + RUNPOD, REPLICATE, FIREWORKS keys
│   └── prod.py            # Prod DB + same API keys
├── users/
│   ├── models.py          # User (email, phone, role, region, district, ward, street, is_verified, etc.)
│   ├── backends.py        # EmailBackend (email-based login)
│   ├── serializers.py     # UserRegistrationSerializer, MyTokenObtainPairSerializer, ChangePasswordSerializer
│   ├── urls.py            # Dashboard, login, register, JWT, staff, verify_phone, location APIs, password_reset
│   └── views.py           # Auth, registration, phone verification (Beem SMS), location (mtaa)
├── ai4chapp/
│   └── urls.py            # path '' -> login view (landing)
├── news/
│   ├── models.py          # News (title, thumbnail, body CKEditor, status, publish, slug, publisher, thematic_area)
│   ├── admin.py           # News registered in admin
│   ├── urls.py            # List, add, review, publish, delete, read by slug, API
│   └── views.py           # NewsAPIView (GET/POST API + classmethods for web views)
├── reports/
│   ├── views.py           # report_view, download_csv_report, analytics, result_detail
│   └── urls.py            # reports/, download/, analytics/, view/<result_id>/<crop_type>/<result_type>
├── beans/ | cassava/ | maize/ | rice/   # Crop apps (same layout)
│   ├── models.py          # CropData, CropPredictionResult, CropDetectionResult
│   ├── views.py           # Web classifier/detector + API view (e.g. BeansDetectAPI)
│   ├── urls.py            # Web routes (e.g. image-beans-disease-classifier, -detector, video-...)
│   ├── api_urls.py        # Detection API (mounted at /<crop>/detect/)
│   ├── forms.py           # UploadForm, multi-file
│   ├── serializers.py     # ImageSerializer, FileSerializer (file, user_id)
│   ├── img_loc.py         # EXIF + reverse geocoding (duplicated per app)
│   └── admin.py           # Optional; beans admin is empty
├── templates/
│   ├── backend/           # Dashboard
│   │   ├── includes/      # base.html, header.html, footer.html
│   │   └── pages/         # login, register, admin (dashboard), news, reports, analytics, staff, verify_phone, etc.
│   └── interfaces/       # Per-crop public UI
│       ├── includes/      # footer, header, master
│       └── beans/ | cassava/ | maize/ | rice/   # e.g. beans-classification.html, beans-detection.html
├── static/                # Source CSS, JS, images; backend/ and interfaces/<crop>/
├── staticfiles/          # collectstatic output (do not edit)
├── locale/                # en/, sw/ for translations
├── media/                 # User uploads (gitignored)
├── requirements.txt
└── README.md
```

**Note:** Views reference `system/users/password/` and `system/pages/` (e.g. password reset email, public news). Ensure those template paths exist (e.g. under `templates/system/`) or add them.

### Crop app layout (beans, cassava, maize, rice)

Each crop app has:

- **models.py** — `CropData`, `CropPredictionResult`, `CropDetectionResult` (upload paths relative to MEDIA_ROOT or app).
- **views.py** — Web views (classifier/detector pages) + API view (e.g. `BeansDetectAPI`) using YOLO/Keras.
- **urls.py** — Web routes (e.g. `image-beans-disease-classifier`, `image-beans-disease-detector`, `video-beans-disease-detector`).
- **api_urls.py** — Single detection endpoint; mounted at `/<crop>/detect/` in root `urls.py` (outside i18n).
- **forms.py** — e.g. `UploadForm`, `MultipleFileField`.
- **serializers.py** — `ImageSerializer`, `FileSerializer(file, user_id)`.
- **img_loc.py** — `extract_image_location(image_file)` and helpers (duplicated; see Reusable modules).
- **admin.py** — Optional; crop models are not registered in admin by default.

---

## Key Components

### 1. User model (`users`)

- **Model:** `User` (extends AbstractUser), `USERNAME_FIELD = 'email'`, `REQUIRED_FIELDS = ['username', 'phone_number', 'region', 'district', 'ward', 'street']`.
- **Fields:** email (unique), phone_number (+255, unique), region, district, ward, street, role (admin, manager, agrovet, normal), is_verified, verification_code, status. Table name: `users`.
- **Auth backends:** `users.backends.EmailBackend` first, then `ModelBackend`.

### 2. Crop data and results (beans, cassava, maize, rice)

- **Data model** (e.g. `BeansData`): file_id, file_path, file_name, latitude, longitude, region, district, country, full_address, uploaded_by (FK User), upload_date. Upload dirs: e.g. `beans/files`, `beans/beans_predictions`, `beans/beans_detections`, `beans/beans_detection_outputs` (paths in models use `BASE_DIR`; Django resolves relative to `MEDIA_ROOT`).
- **PredictionResult:** user, result_id, file_name, file_path, predicted_disease (choices per crop), confidence_score, probabilities (JSON), created_at.
- **DetectionResult:** user, result_id, file_name, file_path, output_path, file_type (image/video), detection_results (JSON), created_at.

### 3. ML pipeline (per crop)

- **Classification:** Keras `load_model` for `models/classification/<crop>_classification.h5`; image preprocessing and prediction in view.
- **Detection:** Ultralytics `YOLO('models/detection/<crop>_detection.pt')` for image and video (frame-by-frame); response includes boxes (cls, conf, xyxy, etc.), names, orig_shape.
- **Location:** Optional EXIF + reverse geocoding via `img_loc.extract_image_location`; used to fill region, district, country, etc., on Data models.
- **SMS:** Beem Africa used for verification (users) and detection results (crop views); config from `config.base` (BEEM_SMS_API_KEY, BEEM_SMS_SECRET_KEY).

### 4. REST API (detection)

- **Endpoints:** `POST /beans/detect/`, `/cassava/detect/`, `/maize/detect/`, `/rice/detect/` (no language prefix; defined outside `i18n_patterns`).
- **Auth:** Detection API uses `AllowAny`; body: `file`, `user_id`.
- **Behavior:** Saves file to CropData, runs YOLO (or classification where implemented), returns JSON (e.g. `results` with boxes, names, orig_shape).

### 5. Internationalization (i18n)

- **Languages:** English (`en`), Swahili (`sw`). `LANGUAGE_CODE = 'sw'`. `LOCALE_PATHS = [BASE_DIR / 'locale']`.
- **URLs:** Most app routes are under `i18n_patterns` (language-prefixed). Detection APIs and `i18n/` are not. Use `path('i18n/', include('django.conf.urls.i18n'))` for language switching.

---

## Authentication & Users

### Login and session

- **Landing:** `path('', include('ai4chapp.urls'))` → login view. Another login at `users/login/`.
- **After login:** `LOGIN_REDIRECT_URL = 'users:dashboard'`. Dashboard shows counts (news, published news, beans/cassava/maize/rice uploads for current user).
- **Session:** `request.session['user_id']` used in views; logout clears session.

### Registration and phone verification

- **Registration:** `UserRegistrationSerializer`; requires region, district, ward, street (from **mtaa**). On success, sends verification SMS (Beem) and redirects to verify_phone.
- **Phone verification:** Code stored in **Django cache** (key `verification_code_{phone_number}`, 5 min TTL). Resend: `resend_code/<encrypted_phone>/` (phone encrypted with **Fernet**, `FARNET_ENCRYPTION_KEY`). Verify: POST code; on success redirect to login.
- **Beem SMS:** `send_verification_sms(phone_number)` uses Beem API; source_addr e.g. `CROP HEALTH` ( for detection SMS in crop views).

### JWT (API)

- **Obtain token:** `POST /users/login_token/` (MyTokenObtainPairView; custom serializer adds email, role to token).
- **Refresh:** `POST /users/token/refresh/` (TokenRefreshView). Simple JWT: access 5 min, refresh 50 days, rotate refresh, blacklist after rotation.
- **DRF default:** In `settings.py`, the **second** `REST_FRAMEWORK` block wins: `SessionAuthentication`, `IsAuthenticated`. So DRF browsable API uses session; JWT is for explicit API clients.

### Staff management

- **Add staff:** `add_staff` (form with regions from mtaa). **List:** `staff_list`. **Delete:** `delete_staff/<id>`. **Deactivate:** `deactivate_staff/<id>`.

### Password reset

- **Request:** `users/password_reset/` — custom view; sends email with link (uidb64, token) using templates `system/users/password/password_reset_email.txt` and `.html`.
- **Done/Complete:** Django’s `password_reset_done` and `password_reset_complete` are in root urls (under i18n). **Confirm view** (link in email: set new password) is **not** in `urls.py` — add `PasswordResetConfirmView` if you want a full reset flow.

---

## News & Reports

### News

- **Model:** title, thumbnail (ResizedImageField), body (CKEditor RichTextUploadingField), status, publish, reject, thematic_area, slug, publisher (FK User), created_at, updated_at.
- **Workflow:** Add (status=1) → review → publish (publish=1). Public list: `News.objects.filter(publish=1, status=1)`. Read by slug; templates: `system/pages/news.html`, `read_new.html`.
- **Admin:** News is registered in Django admin.
- **API:** `GET/POST /news/api/news` (NewsAPIView).

### Reports

- **report_view:** Aggregates all four crops’ Data + DetectionResult (by file_id/result_id), paginated (10 per page). Renders `backend/pages/reports.html`.
- **download_csv_report:** CSV export: ID, email, user region/district, file type, crop type, image region/district, date, plus one column per detection class (counts).
- **analytics:** Time filter (day / week / month / all). Shows: users per region, crop distribution (counts), prediction counts per crop, file type distribution. Template: `backend/pages/analytics.html`.
- **result_detail:** View a single prediction or detection: `reports/view/<result_id>/<crop_type>/<result_type>` (crop_type: beans|cassava|maize|rice; result_type: prediction|detection). Renders `backend/pages/result_detail.html`.

---

## Settings Reference

### File upload

- `DATA_UPLOAD_MAX_MEMORY_SIZE = 104857600` (100 MB)
- `FILE_UPLOAD_MAX_MEMORY_SIZE = 104857600` (100 MB)

### Static and media

- `STATIC_URL = '/static/'`, `STATICFILES_DIRS = (BASE_DIR / 'static',)`, `STATIC_ROOT = BASE_DIR / 'staticfiles'`
- `STATICFILES_STORAGE = 'whitenoise.storage.CompressedStaticFilesStorage'`
- `MEDIA_ROOT = BASE_DIR / 'media'`, `MEDIA_URL = '/media/'`

### REST framework

- **Effective config** (second block in settings): `DEFAULT_AUTHENTICATION_CLASSES = [SessionAuthentication]`, `DEFAULT_PERMISSION_CLASSES = [IsAuthenticated]`. Pagination and JWT are defined but overridden by this block for default behavior.
- **Simple JWT:** Access 5 min, refresh 50 days; rotate refresh; blacklist after rotation; `AUTH_HEADER_TYPES = ('Bearer',)`.

### django-resized

- Default size [1920, 1080], scale 0.5, quality 75, keep meta, force JPEG, normalize rotation.

### TinyMCE

- `TINYMCE_JS_URL` points to CDN (no-api-key). `TINYMCE_DEFAULT_CONFIG`: silver theme, plugins/toolbars for text, tables, media, code, etc.

### CKEditor

- `CKEDITOR_UPLOAD_PATH = 'uploads/'`, `CKEDITOR_IMAGE_BACKEND = 'pillow'`. Upload/browse views are `login_required`; URLs: `/ckeditor/upload/`, `/ckeditor/browse/`.

### CORS

- `django-cors-headers` is in `MIDDLEWARE`. No `CORS_ALLOWED_ORIGINS` etc. in settings (defaults apply).

### Middleware order

1. SecurityMiddleware  
2. WhiteNoiseMiddleware  
3. SessionMiddleware  
4. LocaleMiddleware  
5. CommonMiddleware  
6. CsrfViewMiddleware  
7. AuthenticationMiddleware  
8. MessageMiddleware  
9. XFrameOptionsMiddleware  
10. CorsMiddleware  
11. AccountMiddleware (allauth)  

---

## Reusable Modules & Consistency

### 1. Crop app template

- Copy one crop app (e.g. beans); keep same files (models, views, urls, api_urls, forms, serializers, img_loc).
- Change model names, DISEASE_CHOICES, upload paths, and ML paths (`models/classification/`, `models/detection/`).
- Register app in `INSTALLED_APPS` and add web + `/<crop>/detect/` URLs in `ai4ch/urls.py`.
- Add `templates/interfaces/<crop>/` and `static/interfaces/<crop>/` as needed.

### 2. img_loc (EXIF + geocoding)

- **Current:** Same logic in `beans/img_loc.py`, `cassava/img_loc.py`, `maize/img_loc.py`, `rice/img_loc.py`.
- **Recommendation:** Move to shared package (e.g. `core/img_loc.py`) with `extract_image_location(image_file)`, `get_location_from_coords(lat, lon)`, and coordinate helpers. Add `exifread` and `geopy` to `requirements.txt`. Replace imports in all four crop apps.

### 3. Data / result models

- Keep schema: Data (file + geo + user), PredictionResult (classification), DetectionResult (detection + JSON). Naming: `CropData`, `CropPredictionResult`, `CropDetectionResult`; related_name `'<crop>_predictions'` / `'<crop>_detections'`.

### 4. API pattern

- Detection: `APIView`, `MultiPartParser`, `FormParser`, `FileSerializer(file, user_id)`, run YOLO, return consistent JSON. Mount at `/<crop>/detect/` outside i18n.

### 5. Config and frontend

- Keep env in `config/`; no secrets in `settings.py`. Use `.env` and document variables. Frontend: `backend/` for dashboard, `interfaces/<crop>/` for crop UIs; Crispy Forms (Bootstrap 4); TinyMCE/CKEditor for rich text.

---

## Configuration & Environment

### Environment variables (`.env`)

- **Base:** `SECRET_KEY`, `FROM_EMAIL_ADDRESS`, `FROM_EMAIL_ADDRESS_PASSWORD`, `FARNET_ENCRYPTION_KEY`, `BEEM_SMS_API_KEY`, `BEEM_SMS_SECRET_KEY`
- **Dev:** `DB_HOST_DEV`, `DB_NAME_DEV`, `DB_USER_DEV`, `DB_PASSWORD_DEV`, `DB_PORT_DEV`, `RUNPOD_API_KEY`, `REPLICATE_API_TOKEN`, `FIREWORKS_API_KEY`
- **Prod:** `DB_HOST_PROD`, `DB_NAME_PROD`, `DB_USER_PROD`, `DB_PASSWORD_PROD`, `DB_PORT_PROD`, plus any API keys

### Switching environments

- In `ai4ch/settings.py`, DB is taken from `config.dev` (or switch to `config.prod`). Set `DEBUG` and `ALLOWED_HOSTS`/`CSRF_TRUSTED_ORIGINS` for production (e.g. `https://portal.ai4crophealth.or.tz`).

### Media and ML models

- **Media:** `media/` is gitignored; ensure it exists and is writable.
- **ML models:** Each crop expects e.g. `models/classification/<crop>_classification.h5` and `models/detection/<crop>_detection.pt` under the app directory (or as referenced in views). `.gitignore` includes `models/`; document how to obtain or build these files for deployment.

---

## Getting Started

1. **Clone and virtualenv**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install ultralytics exifread geopy mtaa   # if not present
   ```
   For Keras `.h5` models, TensorFlow is pulled in via `tensorflow_hub` (or install `tensorflow` explicitly).

3. **Environment**
   - Create `.env` from the variables listed in [Configuration & Environment](#configuration--environment). There is no `.env.example` in the repo; add one if helpful.

4. **Database**
   - Create MySQL database, then:
   ```bash
   python manage.py migrate
   python manage.py createsuperuser
   ```

5. **Static files**
   ```bash
   python manage.py collectstatic --noinput
   ```

6. **Run server**
   ```bash
   python manage.py runserver
   ```

7. **Optional**
   - Add crop ML models (`.h5`, `.pt`) under the paths expected by each app’s views.
   - Add `templates/system/` for password reset and public news if missing (see [Important Notes & Gaps](#important-notes--gaps)).
   - Compile messages: `python manage.py compilemessages` if you use `.po`/`.mo` in `locale/`.

---

## URL Structure

| Path | Description |
|------|-------------|
| `/` | Login (ai4chapp landing) |
| `/<lang>/` | Same as below with language prefix (e.g. `/en/`, `/sw/`) |
| `/users/login/` | Login form |
| `/users/dashboard/` | Dashboard (counts) |
| `/users/register/` | Registration (mtaa regions) |
| `/users/verify_phone/` | Phone verification |
| `/users/login_token/`, `/users/token/refresh/` | JWT obtain/refresh |
| `/users/password_reset/` | Password reset request |
| `/users/get-districts/`, `get-wards/`, `get-streets/` | Location APIs (mtaa) |
| `/users/add-staff/`, `staffs/`, etc. | Staff management |
| `/news/` | Public news list (publish=1, status=1) |
| `/news/news-list/`, `add-new/`, `review-new`, `publish-new`, etc. | Backend news workflow |
| `/news/<slug>/` | Read single news |
| `/reports/reports/`, `download/`, `analytics/` | Reports and CSV |
| `/reports/view/<result_id>/<crop_type>/<result_type>` | Result detail |
| `/beans/`, `/cassava/`, `/maize/`, `/rice/` | Crop classifier/detector pages |
| `/beans/detect/`, `/cassava/detect/`, `/maize/detect/`, `/rice/detect/` | REST detection API (no i18n prefix) |
| `/admin/`, `/auth-api/` | Django admin, DRF browsable API |
| `/accounts/` | django-allauth (e.g. Google) |
| `/i18n/` | Language switcher |
| `/tinymce/`, `/ckeditor/upload/`, `/ckeditor/browse/` | Rich text editors |
| `/password_reset/done/`, `/reset/done/` | Password reset done/complete (under i18n) |

---

## Important Notes & Gaps

1. **Password reset:** The view that handles the email link (uidb64 + token) to set a new password (`PasswordResetConfirmView`) is not in `urls.py`. Add it (e.g. `path('reset/<uidb64>/<token>/', ...)`) for a complete flow.
2. **Templates:** Code references `system/users/password/` and `system/pages/` (e.g. `news.html`, `read_new.html`). If these are missing, add `templates/system/` with the expected structure.
3. **Admin:** Only News is registered. Optionally register crop Data/Prediction/Detection models in respective `admin.py` for easier inspection.
4. **Detection API auth:** Endpoints use `AllowAny`. Consider requiring JWT or session for production.
5. **REST_FRAMEWORK:** Two blocks in settings; the second overrides the first. Default DRF auth is session; JWT is used only where explicitly configured (e.g. login_token).
6. **Dependencies:** Add `ultralytics`, `exifread`, `geopy`, `mtaa` to `requirements.txt` (or a dev/prod split) so installs are reproducible.
7. **Reports:** Aggregation matches Data to DetectionResult; ensure `file_id` / `result_id` semantics match (e.g. if they are the same value when a detection is saved) so CSV and report view are correct.

This README should give you a full picture of the technology stack, architecture, key components, reusable patterns, and gaps to address for consistency and production readiness.
