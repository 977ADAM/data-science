# Customer Churn + Uplift API (Telco)

ML‑сервис для оценки оттока клиентов (churn), uplift‑моделирования удерживающих воздействий, A/B‑таргетинга и мониторинга дрейфа данных. Реализован как production‑ориентированный FastAPI‑сервис с версионированием моделей и manifest‑контролем целостности.

---

## 1. Назначение проекта

Проект решает задачу **управления удержанием клиентов**:

1. Оценивает вероятность оттока клиента (churn probability)
2. Оценивает uplift удерживающего воздействия (если обучена uplift‑модель)
3. Позволяет отбирать клиентов для retention‑кампаний (A/B: churn vs uplift)
4. Мониторит дрейф входных данных и предсказаний модели

---

## 2. Основные возможности

### 2.1 Churn prediction

Для каждого клиента рассчитывается:

P(churn = 1 | customer features)

Используется sklearn Pipeline:
- очистка и нормализация данных
- feature engineering
- One‑Hot Encoding категориальных признаков
- CatBoostClassifier
- пост‑калибровка вероятностей (Isotonic)

Порог классификации выбирается **по максимизации бизнес‑метрики Expected Value**, а не фиксированно 0.5.

---

### 2.2 Uplift modeling (опционально)

Поддерживается uplift‑скоринг:

- p_treated = P(Y=1 | X, do(treatment=1))
- p_control = P(Y=1 | X, do(treatment=0))
- uplift = p_treated − p_control

Доступные алгоритмы:
- T‑Learner (по умолчанию)
- S‑Learner
- X‑Learner
- DR‑Learner

---

### 2.3 Drift monitoring

Для батча входящих клиентов рассчитывается:

- Data drift
  - numeric: PSI, KS‑stat
  - categorical: PSI, L1 distance, unseen category mass
- Prediction drift
  - PSI и KS по распределению churn probabilities

Статусы:
- PSI ≤ 0.2 — OK
- 0.2 < PSI ≤ 0.3 — WARNING
- PSI > 0.3 — CRITICAL

---

## 3. Данные

### 3.1 Источник данных для обучения

По умолчанию используется Telco датасет:

- `Churn.csv`

Целевая переменная:
- `Churn` (Yes / No → 1 / 0)

---

### 3.2 Входные признаки клиента (API)

API принимает JSON с описанием одного клиента.

**Числовые признаки:**
- SeniorCitizen (0/1)
- tenure
- MonthlyCharges
- TotalCharges

**Категориальные признаки:**
- gender
- Partner
- Dependents
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- PaperlessBilling
- Contract
- PaymentMethod

Допускается:
- отсутствие части полей
- строковый формат TotalCharges
- неизвестные категории

Pipeline автоматически выравнивает и очищает данные.

---

## 4. Feature engineering

Автоматически добавляются признаки:

- avg_monthly_bill = TotalCharges / (tenure + 1)
- high_charges
- is_new_client (tenure < 12)
- is_long_term (tenure > 24)
- tenure_bucket
- num_services
- is_month_to_month
- is_auto_pay
- revenue_per_tenure

---

## 5. Обучение модели

### 5.1 Churn

Запуск:

```bash
python -m src.train_pipeline
```

Во время обучения:
- train/test split
- обучение pipeline
- калибровка вероятностей
- расчёт ROC‑AUC
- подбор оптимального threshold по Expected Value

---

### 5.2 Uplift

Запуск:

```bash
python -m src.train_uplift
```

Ожидается колонка treatment (0/1).
Если treatment отсутствует — модель деградирует к базовому сценарию.

---

## 6. Версионирование и manifest

Каждая модель сохраняется в:

```
models/<MODEL_VERSION>/
```

Сохраняется manifest.json с:
- sha256 данных
- sha256 кода
- sha256 модели
- reference distributions для drift

Поддерживается:
- models/latest
- models/latest.txt

При несовпадении hash — сервис отказывается обслуживать предсказания.

---

## 7. API

### 7.1 GET /health

Проверка состояния сервиса и загруженных моделей.

---

### 7.2 POST /predict

Churn‑скоринг одного клиента.

**Request:**
```json
{
  "customer": {
    "tenure": 5,
    "MonthlyCharges": 95.2,
    "TotalCharges": "475",
    "Contract": "Month-to-month"
  }
}
```

**Response:**
```json
{ "churn_probability": 0.73 }
```

---

### 7.3 POST /uplift

Uplift‑оценка одного клиента.

**Response:**
```json
{
  "p_treated": 0.40,
  "p_control": 0.55,
  "uplift": -0.15
}
```

---

### 7.4 POST /drift

Drift‑анализ батча клиентов.

**Request:**
```json
{
  "customers": [ { ... }, { ... } ]
}
```

Возвращает детальную структуру data drift и prediction drift.

---

### 7.5 POST /ab/select

A/B таргетинг:

- control — top‑K по churn probability
- treatment — top‑K по uplift

---

### 7.6 GET /metrics

Метрики в формате Prometheus:
- churn_avg_proba
- churn_psi_max
- churn_psi_status

---

## 8. Конфигурация (env)

- MODEL_VERSION
- UPLIFT_MODEL_VERSION
- REQUIRE_MANIFEST
- UPLIFT_LEARNER
- TREATMENT_COL
- LTV_SAVED
- COST_ACTION
- RETENTION_UPLIFT

---

## 9. Запуск API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

## 10. Структура проекта

- app/main.py — FastAPI endpoints
- app/schemas.py — API схемы
- src/train_pipeline.py — churn обучение
- src/train_uplift.py — uplift обучение
- src/pipeline.py — ML pipeline
- src/preprocessing.py — очистка данных
- src/drift.py — drift мониторинг
- src/versioning.py — manifest и hash‑контроль
- src/config.py — конфигурация

---

Проект готов к production‑использованию и может быть встроен в CRM / retention‑платформу.

