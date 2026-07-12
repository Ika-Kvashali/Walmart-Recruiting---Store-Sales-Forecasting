# Walmart Recruiting — Store Sales Forecasting

**Kaggle კონკურსი:** [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)

პროექტის მიზანია გამოვიკვლიოთ სხვადასხვა არქიტექტურის time series მოდელები — Deep Learning, Tree-Based, Classical Statistical და Foundation Models — და შევადაროთ ერთმანეთს კონკრეტულ ამოცანაზე: 45 Walmart-ის მაღაზიის და ~81 დეპარტამენტის ყოველკვირეული გაყიდვების პროგნოზირება, სულ ~3,300 ცალკეული (Store, Dept) time series.

**შეფასების მეტრიკა:** WMAE (Weighted Mean Absolute Error), სადაც holiday კვირები 5x წონით ითვლება:

```
WMAE = Σ(wᵢ · |yᵢ - ŷᵢ|) / Σ(wᵢ)     wᵢ = 5 თუ holiday week, სხვა შემთხვევაში 1
```

---

## შინაარსი

1. [რეპოზიტორიის სტრუქტურა](#რეპოზიტორიის-სტრუქტურა)
2. [MLflow / W&B ტრეკინგის სტრუქტურა](#mlflow--wb-ტრეკინგის-სტრუქტურა)
3. [საერთო მეთოდოლოგია](#საერთო-მეთოდოლოგია)
4. [მოდელები — აღწერა და გამოცდილება](#მოდელები--აღწერა-და-გამოცდილება)
5. [შედეგების შედარების ცხრილი](#შედეგების-შედარების-ცხრილი)
6. [საბოლოო არჩევანი და Kaggle შედეგი](#საბოლოო-არჩევანი-და-kaggle-შედეგი)
7. [მთავარი დასკვნები (Key Lessons)](#მთავარი-დასკვნები-key-lessons)

---

## რეპოზიტორიის სტრუქტურა

| ფაილი | არქიტექტურა | ტიპი |
|---|---|---|
| `model_experiment_N_BEATS.ipynb` | N-BEATS | Deep Learning |
| `model_experiment_DLinear.ipynb` | DLinear | Deep Learning |
| `model_experiment_TFT.ipynb` | Temporal Fusion Transformer | Deep Learning |
| `model_experiment_TimesFM.ipynb` | TimesFM (Zero-shot + LoRA fine-tune) | Foundation Model |
| `model_experiment_Light_GBM.ipynb` | LightGBM | Tree-Based |
| `model_experiment_XGBoost.ipynb` | XGBoost | Tree-Based |
| `model_experiment_Prophet.ipynb` | Prophet | Classical Statistical |
| `model-experiment-ARIMA.ipynb` | ARIMA | Classical Statistical |
| `model-experiment-SARIMA.ipynb` | SARIMA | Classical Statistical |
| `model_inference.ipynb` | — | საბოლოო submission-ის გენერაცია საუკეთესო რეგისტრირებული მოდელით |

---

## MLflow / W&B ტრეკინგის სტრუქტურა

პროექტის მოთხოვნის მიხედვით, თითოეულ არქიტექტურას აქვს **ცალკე Experiment**, ხოლო თითოეული ეტაპი (preprocessing → feature engineering → feature selection → training) ლოგირებულია, როგორც ცალკე **Run** ამ Experiment-ის შიგნით.

**Deep Learning მოდელები (N-BEATS, DLinear, TFT, TimesFM)** ლოგირებულია **Weights & Biases**-ზე, საერთო პროექტში `Walmart Weekly Sales Prediction`, ცალკე `group`-ებით არქიტექტურის მიხედვით (`entity: ikvas22-free-university-of-tbilisi`):

```
W&B Project: Walmart Weekly Sales Prediction
├── Group: NBEATS
│   ├── NBEATS_Preprocessing
│   ├── NBEATS_Baseline
│   └── NBEATS_HPSearch
├── Group: DLinear   (იგივე სტრუქტურა)
├── Group: TFT       (იგივე სტრუქტურა)
└── Group: TimesFM
    ├── TimesFM_Preprocessing
    ├── TimesFM_ZeroShot
    └── TimesFM_FineTune_LoRA
```

**Tree-Based და Statistical მოდელები (LightGBM, XGBoost, Prophet, ARIMA, SARIMA)** ლოგირებულია **MLflow**-ზე (DagsHub-ის საშუალებით: `dagshub.init(repo_owner='ikvas22', repo_name='Walmart-Recruiting---Store-Sales-Forecasting', mlflow=True)`):

```
MLflow Experiment: LightGBM_Training
├── exp{N}_{name}_preprocessing
├── exp{N}_{name}_feature_engineering
├── exp{N}_{name}_feature_selection
└── exp{N}_{name}_training

MLflow Experiment: XGBoost_Training      (იგივე სტრუქტურა)
MLflow Experiment: Prophet_Training      (მხოლოდ EDA + Training — იხ. ქვემოთ, რატომ)
MLflow Experiment: ARIMA_tarining        (per-series rolling-CV, 3 ექსპერიმენტი)
MLflow Experiment: SARIMA_training       (per-series rolling-CV, 3 ექსპერიმენტი)
```

ყველა არქიტექტურის საუკეთესო მოდელი რეგისტრირებულია MLflow Model Registry-ში:

| Registry Name | წყარო |
|---|---|
| `nbeats_walmart_best` | N-BEATS |
| `dlinear_walmart_best` | DLinear |
| `tft_walmart_best` | TFT |
| `timesfm_walmart_best` | TimesFM |
| `Light_GBM_Training_Best` | LightGBM |
| `XGBoost_Best` | XGBoost |
| `prophet_walmart_best` | Prophet — **საბოლოოდ არჩეული მოდელი** |
| `ARIMA_Walmart_SalesForecast` | ARIMA |
| `SARIMA_Walmart_SalesForecast` | SARIMA |

---

## საერთო მეთოდოლოგია

### მონაცემები
`train.csv`, `test.csv`, `features.csv`, `stores.csv` — Kaggle-ის ოფიციალური მონაცემები. `train`+`features`+`stores` merge-დება `(Store, Date, IsHoliday)` და `Store` key-ებზე.

### Train/Validation split
`VAL_START = "2012-04-01"` — ყველა მოდელი ამ თარიღით ჰყოფს train/val-ს.

### Feature Engineering (Tree-Based მოდელებისთვის)
- **Lags:** `[1, 2, 3, 4, 13, 26, 52]` კვირა
- **Rolling windows:** `[4, 12, 26, 52]` კვირა (mean/std/min/max/median/skew + EWMA)
- **Calendar features:** week/month/quarter, sin/cos ციკლური encoding, კონკრეტული holiday flags
- **Feature Selection:** სამსაფეხურიანი ფილტრი — WOE/IV → Mutual Information → Correlation (threshold-ები: `IV=0.02`, `MI=0.05`, `Corr=0.90`)

**Deep Learning მოდელები (N-BEATS, DLinear, TFT, TimesFM) არ იყენებენ ამ ინჟინერულ feature-ებს** — ისინი ცალკეული (raw) time series-ზე მუშაობენ პირდაპირ (`unique_id`, `ds`, `y`), TFT-ის გარდა, რომელიც დამატებით იყენებს covariate-ებს (MarkDown, Temperature, CPI და ა.შ.).

---

## მოდელები — აღწერა და გამოცდილება

### N-BEATS

**მოკლე აღწერა:** სუფთა MLP-ზე დაფუძნებული არქიტექტურა, ყოველგვარი attention/recurrence გარეშე. Interpretable ვარიანტში ცალკე stack-ები სწავლობენ trend-ს (polynomial basis) და seasonality-ს (Fourier basis).

**ჩვენი გამოცდილება:**
- გამოვიყენეთ `neuralforecast` ბიბლიოთეკა; Colab-ზე მუდმივი ვერსიების კონფლიქტი იყო `torch`/`numpy`/`torchvision` შორის — საბოლოოდ დავფიქსირდით `torch==2.2.0`, `numpy<2`, `neuralforecast==1.7.4` კომბინაციაზე.
- `AutoNBEATS` + Ray Tune-ის ავტომატური search არასტაბილური აღმოჩნდა (`n_blocks` სერიალიზაციის ბაგი) — გადავედით **manual search**-ზე, იგივე შედეგით, მაგრამ სრული კონტროლით.
- საწყისი evaluation მხოლოდ 4-კვირიან ერთ ფანჯარაზე მუშაობდა (`WMAE == MAE` ზუსტად ემთხვეოდა — ნიშანი, რომ holiday კვირა საერთოდ არ ხვდებოდა val-ში). გამოსწორდა `evaluate_cv` (rolling, 7 fold) დანერგვით.
- **დასკვნა:** hyperparameter search-მა (15+ trial) პრაქტიკულად ვერ გააუმჯობესა hand-picked baseline — თანმხვედრია N-BEATS-ის ორიგინალურ ნაშრომთან, სადაც აღნიშნულია მოდელის დამოკიდებულება hyperparameter-ების მიმართ.

### DLinear

**მოკლე აღწერა:** Decomposition + ორი Linear layer (trend/seasonality). 2023 წლის ICLR ნაშრომის თანახმად, ხშირად სჯობია რთულ Transformer-ებს time series-ზე.

**ჩვენი გამოცდილება:** იგივე pipeline-ი და bug-fix-ები, რაც N-BEATS-ს (rolling-origin CV, `start_padding_enabled=True`). Search space: `moving_avg_window` — trend-ის ამოღების ფანჯარა.

### Temporal Fusion Transformer (TFT)

**მოკლე აღწერა:** ერთადერთი DL მოდელი პროექტში, რომელიც იყენებს covariate-ებს — `hist_exog` (MarkDown, Temperature, CPI), `futr_exog` (calendar features, ცნობილი წინასწარ). Static exog (store Type/Size) მოვხსენით სირთულის შესამცირებლად.

**ჩვენი გამოცდილება:**
- `evaluate_cv` ფუნქცია საჭიროებდა სრული covariate frame-ის გადაცემას, არა მხოლოდ `(unique_id, ds, y)`.
- ყველაზე მძიმე DL მოდელი train-ისთვის, მაგრამ validation-ზე ვერ გადააჭარბა N-BEATS-ს — სავარაუდო მიზეზი: MarkDown feature-ები ~65%-ით ცარიელია, ასუსტებს covariate-ის უპირატესობას.

### TimesFM

**მოკლე აღწერა:** Google-ის pretrained Foundation Model time series-ისთვის (decoder-only Transformer, patch-based). გატესტილია ორივე რეჟიმში: **Zero-shot** (patch-ის pretrained checkpoint-ით პირდაპირ) და **LoRA fine-tune** (HuggingFace Transformers + PEFT).

**ჩვენი გამოცდილება:**
- pip-ზე ხელმისაწვდომი ვერსია (`timesfm==2.0.2`) რეალურად შეიცავს **TimesFM 2.5**-ის API-ს, რომელიც მნიშვნელოვნად განსხვავდება საბაზისო დოკუმენტაციისგან (`TimesFm` კლასის ნაცვლად `TimesFM_2p5_200M_torch.from_pretrained()`).
- Zero-shot evaluation თავდაპირველად ძალიან ნელი იყო (ერთი series ერთი forecast call) — დაჩქარდა batch-ური `forecast()` გამოძახებით.
- LoRA fine-tuning მოითხოვდა **სხვა checkpoint-სა და model class-ს** (`TimesFm2_5ModelForPrediction` HuggingFace-იდან, არა pip პაკეტის `timesfm` მოდული) — official example-ის (`google-research/timesfm` repo) გაცნობის შემდეგ დავამატეთ სწორად.
- გამუდმებით ვხვდებოდით torch ecosystem-ის ვერსიების კონფლიქტებს (`torchvision`, `torchaudio`, `torchao`) — თითოეული cleanup-ის შემდეგ საჭირო იყო runtime-ის factory reset.
- **Fine-tuned ვერსია საბოლოოდ სჯობდა zero-shot-ს** validation-ზე.

### LightGBM

**მოკლე აღწერა:** Gradient Boosting მოდელი ინჟინერული feature-ების სრულ ნაკრებზე (lags, rolling stats, calendar, markdown, store-level feature-ები).

**ჩვენი გამოცდილება:**
- თავდაპირველ ვერსიაში `GridSearchCV` + `TimeSeriesSplit(n_splits=3)` იყენებდა row-index-ზე დაფუძნებულ split-ს, რომელიც არ იყო ნამდვილად ქრონოლოგიური (მონაცემები დალაგებულია `Store, Dept, Date`-ით, არა მხოლოდ `Date`-ით).
- გადავაკეთეთ **Option A**-ზე: rolling-origin CV, სადაც ყოველ fold-ზე ხელახლა ვაწყობთ feature pipeline-საც და feature selector-საც (არა მხოლოდ მოდელს) — რომ თავიდან ავიცილოთ feature-selection-ის leakage.
- `mlflow.lightgbm.log_model()` რამდენჯერმე **მდუმარედ ვერ ატვირთა** მოდელის artifact DagsHub-ზე (მხოლოდ `param_grid.json` ჩანდა server-ზე). გამოსწორდა ლოკალურად `save_model` + ფაილ-ფაილზე ატვირთვით (`log_artifact`), ყოველთვის ვერიფიცირებით `MLmodel`-ის არსებობაზე server-ზე რეგისტრაციამდე.

### XGBoost

**მოკლე აღწერა:** LightGBM-ის მსგავსი Gradient Boosting მოდელი, იგივე feature ნაკრებზე.

**ჩვენი გამოცდილება:** იდენტური ბაგები და გამოსწორებები, რაც LightGBM-ს (`TimeSeriesSplit` → rolling-origin CV Option A). MLflow registration-ში დამატებულია robust artifact-path detection და `champion` alias/stage-transition fallback.

### Prophet — ⭐ საბოლოოდ არჩეული მოდელი

**მოკლე აღწერა:** Meta-ს decomposable სტატისტიკური მოდელი — trend + yearly seasonality + holiday effects, თითოეულ (Store, Dept) series-ზე ცალკე გაწყობით.

**ჩვენი გამოცდილება:**
- Prophet fit-ავს **ერთ მოდელს თითო series-ზე** — 3,300 series-ის სრულად დამუშავება მაღალი grid-ით არარეალურია დროში, ამიტომ scope-ი შემოვსაზღვრეთ top-N series-ით (sales-ის მოცულობით), 600-დან 1,200-მდე და საბოლოოდ ექსპერიმენტების შედეგების მიხედვით ავირჩიეთ ერთი ჰიპერპარამეტრების კრებული, რომელიც 3,000 series-ზე გამოვცადეთ, რამაც საუკეთესო შედეგი მოგვცა.
- Hyperparameter grid მოიცავდა `changepoint_prior_scale`, `seasonality_mode`, `holidays_prior_scale` (holiday-ეფექტების წონა — WMAE-ის 5x holiday weighting-ის გამო განსაკუთრებით მნიშვნელოვანი).
- MLflow სტრუქტურაში Feature Engineering და Feature Selection Run-ები **განზრახ გამოტოვებულია** — Prophet მხოლოდ ნედლ `(ds, y)` სერიასა და holiday calendar-ს მოიხმარს, ასე რომ ეს ეტაპები ცარიელი იქნებოდა.
- **Wrapper v1 → v2:** თავდაპირველი registry wrapper 4 კვირას წინასწარ განსაზღვრული horizon-ით პროგნოზირებდა (`H=4`), რაც Kaggle-ის ~39-კვირიან test set-ზე კატასტროფულად არასწორ Id-ებს აწარმოებდა. გადავაკეთეთ **v2**-ზე, რომელიც ნედლ train + test (`Weekly_Sales=NaN`) მონაცემებს იღებს და ზუსტად მოთხოვნილ თარიღებს პროგნოზირებს ნებისმიერი horizon-ით — ეს არის საბოლოო inference-ში გამოყენებული ვერსია.

### ARIMA

**მოკლე აღწერა:** კლასიკური (non-seasonal) ARIMA, per-series. სეზონურობის დასაფარად გამოყენებულია გარეგანი Fourier terms (`order=3`, `period=52`), რადგან ARIMA-ს საკუთარი seasonal კომპონენტი არ გააჩნია.

**ჩვენი გამოცდილება (გუნდის წევრის მიერ):** rolling-origin CV (`CV_FOLDS=3`, `H=4`), 3 ექსპერიმენტული კონფიგურაცია (auto_arima-ს stepwise search სხვადასხვა Fourier order-ითა და `arima_kwargs`-ით). პროფესორის მითითებით, ARIMA/SARIMA უფრო თეორიული შედარებისთვისაა საჭირო — ტრენინგზე დიდი დრო არ დახარჯულა.

### SARIMA

**მოკლე აღწერა:** ARIMA-ს გაფართოება საკუთარი სეზონური კომპონენტით `(P, D, Q, m=52)`. ამიტომ, ARIMA-სგან განსხვავებით, არ საჭიროებს გარეგან Fourier terms-ს — სეზონურობას თავად სეზონური კომპონენტი ფარავს.

**ჩვენი გამოცდილება (გუნდის წევრის მიერ):** იგივე rolling-origin CV სტრუქტურა და 3-ექსპერიმენტიანი grid, რაც ARIMA-ს (`seasonal=True, m=52`). სეზონური fit ბუნებრივად უფრო ნელია `m=52`-ის გამო.

---

## შედეგების შედარების ცხრილი

*ივსება საბოლოო შედეგებით — ცხრილში ერთმანეთისგან განცალკევებულია validation (rolling-origin CV) და საბოლოო Kaggle test score, რადგან, როგორც ქვემოთ არის ახსნილი, ეს ორი რიცხვი მნიშვნელოვნად შეიძლება განსხვავდებოდეს.*

| მოდელი | არქიტექტურის ტიპი | Rolling-CV WMAE | Kaggle Public WMAE | შენიშვნა |
|---|---|---|---|---|
| N-BEATS | Deep Learning | 1360.67 | _ | **საუკეთესო DL მოდელი** |
| DLinear | Deep Learning | 1712.12 | _ | |
| TFT | Deep Learning | 1722.65 | _ | **ყველაზე ნელი და უარესი DL მოდელი** |
| TimesFM (Zero-shot) | Foundation Model | ~1500 | _ | |
| TimesFM (LoRA fine-tuned) | Foundation Model | 1361.27 | _ | |
| LightGBM | Tree-Based | 458.25 | ~4,700 | |
| XGBoost | Tree-Based | 510.15 | _ | |
| Prophet | Statistical | 1351.2 | ~4,000 | **საბოლოო არჩევანი** |
| ARIMA | Statistical | 6413.14 | _ | |
| SARIMA | Statistical | ~10k | _ | **დროის გამო ბევრი შეზღუდვაა, underfitted** |

---

## საბოლოო არჩევანი და Kaggle შედეგი

**არჩეული მოდელი: Prophet** (`prophet_walmart_best`, MLflow Model Registry-დან).

**რატომ Prophet, მიუხედავად იმისა, რომ validation-ზე ყველაზე კარგი არ იყო:** იხილეთ ქვემოთ, "მთავარი დასკვნები" — მოკლედ, Prophet-ის ერთჯერადი (non-recursive) prediction ნებისმიერ horizon-ზე უკეთესად ერგებოდა Kaggle-ის 39-კვირიან test task-ს, ვიდრე lag-feature-ზე დაფუძნებული tree მოდელები.

`model_inference.ipynb` წამოიღებს ამ მოდელს პირდაპირ MLflow Model Registry-დან (`models:/prophet_walmart_best/latest`), აწვდის მას ნედლ train history-სა და ნედლ test set-ს (`Weekly_Sales=NaN` სამომავლო რიგებზე), და აგენერირებს `submission.csv`-ს Kaggle-ზე პირდაპირი submission-ისთვის.

**Kaggle-ის საბოლოო ქულა:** Public: 3885.44422 Private: 4089.43081

---

## მთავარი დასკვნები (Key Lessons)

### 1. Validation task-ი უნდა ემთხვეოდეს deployment task-ს

პროექტის ყველაზე მნიშვნელოვანი აღმოჩენა: ჩვენი rolling-origin CV ზომავდა **4-კვირიან** წინსწრებას, ხოლო Kaggle-ის ტესტი მოითხოვდა **~39-კვირიან** წინსწრებას ერთბაშად. ეს ორი სხვადასხვა სირთულის ამოცანაა — ნებისმიერი მოდელის სიზუსტე უარესდება horizon-ის ზრდასთან ერთად, რადგან მომავალი უფრო შორეულ პერიოდში არსებითად ნაკლებად პროგნოზირებადია. ამიტომ validation WMAE-მ **არ იწინასწარმეტყველა** Kaggle-ის რეალური ქულა არცერთი მოდელისთვის.

### 2. Lag-feature მოდელები (LightGBM/XGBoost) განსაკუთრებით მოწყვლადია გრძელ horizon-ზე

LightGBM-ის მთავარი feature-ები (`sales_lag_1`, rolling means) 39-კვირიან test set-ზე პრაქტიკულად ცარიელია (NaN) — Kaggle-ის test set მომავალშია, ნამდვილი sales არ არსებობს. **Recursive prediction**-ით (თითო კვირის პროგნოზი უკან იწერება, როგორც pseudo-actual შემდეგი კვირისთვის) ვხსნით NaN პრობლემას, მაგრამ ვქმნით ახალს: შეცდომები **გამრავლდება feature-ების საშუალებით** — არასწორი პროგნოზი კვირა 10-ზე ხდება არასწორი input კვირა 11-სთვის, და ასე შემდეგ. ეს არის feedback loop, რომელიც კატასტროფულად ამძაფრებს შეცდომას გრძელ horizon-ზე.

### 3. Prophet-ის extrapolation "რბილად" უარესდება, არა კატასტროფულად

Prophet არ იყენებს lag feature-ებს — ის აწყობს trend(t) + seasonality(t) მრუდს, როგორც **თარიღის ფუნქციას**, და უბრალოდ აფასებს ამ მრუდს მომავალი თარიღისთვის. პროგნოზი არასოდეს "მოიხმარს" საკუთარ წინა პროგნოზს — feedback loop არ არსებობს. Horizon-ის ზრდასთან ერთად მხოლოდ trend-ის ექსტრაპოლაციის გაურკვევლობა იზრდება წრფივად, არა რეკურსიულად. სწორედ ამიტომ Prophet-მა, მიუხედავად უარესი validation-ის, **აჯობა** LightGBM-ს Kaggle-ის რეალურ ტესტზე.

### 4. ერთი ფიქსირებული Train/Val split-ი შეიძლება ატყუებდეს

ფიქსირებულმა `VAL_START` split-მა თითქმის ყველა holiday კვირა train-ში ჩასვა, val-ს კი Labor Day-ის გარდა არაფერი დაუტოვა — რაც WMAE-ის (5x holiday weight) გამო არაპროპორციულ და შეუდარებელ `train_wmae`/`val_wmae` წყვილებს წარმოშობდა. Rolling-origin (walk-forward) CV ერთადერთი საიმედო გამოსავალი აღმოჩნდა.

MLflow URL: https://dagshub.com/ikvas22/Walmart-Recruiting---Store-Sales-Forecasting.mlflow/#/
WandB: https://wandb.ai/ikvas22-free-university-of-tbilisi/Walmart%20Weekly%20Sales%20Prediction
