# SprayFrameGen

Генератор синтетических кадров распыла для тестирования алгоритмов компьютерного зрения.

## Состав поставки

- Исходный код программы (`sprayframegen/`)
- Файл запуска `run.bat`
- Зависимости в `requirements.txt` (NumPy 2.3.5, Pillow 12.0.0)
- Документация: `README.md`, `docs/`
- Примеры конфигурации: `docs/examples/config.default.json`, `docs/examples/config.ideal.json`
- Автоматические тесты: `tests/`
- Пример завершённой серии: `docs/examples/series_example/`

## Требования

- Windows 10/11 x64
- Python 3.14 x64 с Tcl/Tk (устанавливается вместе с Python)
- 500 МБ свободного места на диске

## Подготовка окружения (один раз)

### С интернетом

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Без интернета (офлайн)

1. На машине с интернетом скачайте wheels:
   ```powershell
   .venv\Scripts\python.exe -m pip download -d wheels -r requirements.txt
   ```
2. Перенесите папку `wheels` на целевую машину
3. Установите:
   ```powershell
   .venv\Scripts\python.exe -m pip install --no-index --find-links wheels -r requirements.txt
   ```

После подготовки приложение работает полностью офлайн.

## Запуск

```cmd
run.bat [опции]
```

### Основные режимы

| Команда | Описание |
|---------|----------|
| `run.bat` | Открыть окно настроек (GUI) |
| `run.bat --check <файл.json>` | Проверить конфигурацию |
| `run.bat --new <файл.json>` | Создать новую конфигурацию (реалистичный режим) |
| `run.bat --generate <файл.json>` | Сгенерировать серию по конфигурации |
| `run.bat --smoke-test` | Быстрая проверка окружения без GUI |
| `run.bat --environment` | Показать версию окружения и сборки |
| `run.bat --version` | Показать версию программы |

### Примеры

```cmd
rem Проверить пример конфигурации
run.bat --check docs\examples\config.default.json

rem Создать новые настройки
run.bat --new my_settings.json

rem Сгенерировать серию
run.bat --generate my_settings.json

rem Запустить тесты
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Конфигурация

Все настройки хранятся в JSON-файле. Основные группы параметров:

### Серия и камера
- `camera.width_px`, `camera.height_px` — разрешение (640×480, 1024×768, 1280×1024, 1920×1080)
- `camera.frame_count` — число кадров (2–1000)
- `camera.scale_um_per_px` — масштаб, мкм/пиксель
- `camera.frame_interval_us` — межкадровый интервал, мкс
- `camera.exposure_us` — выдержка, мкс (≤ frame_interval_us)
- `camera.format` / `camera.bit_depth` — PNG 8/16 бит, TIFF 16 бит

### Капли
- `droplets.count_per_frame` — капель на кадр (1–10000)
- `droplets.size_distribution` — `lognormal` или `normal`
- Параметры распределения: `median_um`/`geometric_std` или `mean_um`/`std_um`
- `droplets.max_diameter_um` — макс. диаметр (≥ 3 × scale)
- `droplets.concentration` — `low` / `medium` / `high`
- `droplets.render_mode` — `realistic` / `ideal`

### Скорость
- `motion.speed_mean_m_s`, `motion.speed_std_m_s` — модуль скорости
- `motion.speed_min_m_s`, `motion.speed_max_m_s` — границы
- `motion.direction_std_deg` — угловой разброс основной группы
- `motion.motion_blur_enabled` — размытие движения

### Внешний вид
- `appearance.defocus_enabled` — гауссова расфокусировка
- `appearance.blur_sigma_min_px`, `blur_sigma_max_px` — диапазон σ

### Фон и шум
- `background.tone` — `light` (0.85) / `dark` (0.15)
- `background.noise_enabled`, `background.noise_sigma` — сенсорный шум

### Случайная генерация
- `seed` — 32-битное целое (0–4294967295), автогенерация при создании

## Работа через GUI

Окно содержит 8 вкладок:
1. **Серия и камера** — разрешение, кадры, масштаб, выдержка, формат
2. **Капли** — количество, распределение размеров, концентрация, режим
3. **Скорость** — параметры скорости, размытие движения
4. **Внешний вид** — расфокусировка
5. **Фон и шум** — тон фона, шум
6. **Случайная генерация** — seed и кнопка «Новый seed»
7. **Предпросмотр** — чистый кадр + переключатель служебного слоя (ID, контуры, векторы, границы)
8. **Экспорт** — папка, прогресс-бар, время, кнопка «Отмена»

Все сообщения об ошибках — на русском языке с указанием параметра и причины.

## Форматы экспорта

Папка серии: `series_YYYYMMDD_HHMMSS_<seed>`

Внутри:
- `frames/frame_XXXXXX.png` (или `.tif`) — кадры, 6 цифр
- `droplets.csv` — 19 столбцов, UTF-8, LF, `.17g`, `true`/`false`
- `config.json` — полная конфигурация с `run.*` метаданными

### Столбцы CSV
```
frame_number,time_us,droplet_id,is_new,center_x_px,center_y_px,
center_x_mm,center_y_mm,diameter_eq_px,diameter_eq_um,
semi_major_px,semi_minor_px,angle_deg,vx_m_s,vy_m_s,
speed_m_s,blur_sigma_px,border,overlap
```

## Повторяемость

Одинаковые настройки + `seed` + то же окружение = побайтово идентичные кадры и CSV.

- Предпросмотр не расходует состояние экспорта
- Переключение шума не меняет геометрию и CSV
- `config.json` серии содержит `run.environment` для проверки совпадения окружения
- При несовпадении версии/окружения выдаётся предупреждение

## Офлайн-работа

После разовой подготовки окружения:
- Никаких сетевых запросов
- Никакой телеметрии
- Все файлы создаются локально
- Настройки сохраняются в JSON рядом с программой или в указанной папке

## Тестирование

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m unittest tests.ui_smoke -v
```

106 тестов покрывают:
- Конфигурацию и валидацию
- Генерацию капель и геометрию
- Идеальный и реалистичный рендер
- Оптические эффекты (D06)
- Экспорт и CLI
- Контроллер генерации с отменой (D07)
- Полный GUI (D08)
- Приёмочные тесты (D09): 19 критериев ТЗ, статистика, нагрузка 1000×10000

## Пример завершённой серии

В `docs/examples/series_example/` лежит готовая серия с:
- 6 кадрами 640×480 PNG 8 бит
- 100 капель/кадр, реалистичный режим
- `run.status="completed"`, все метаданные
- Побайтово воспроизводится в зафиксированном окружении

## Известные ограничения (v1)

- Только 2D-проекции, нет физического моделирования
- Нет столкновений, коалесценции, дробления
- Нет 3D, нет RGB, нет видео
- Нет интеграции с анализатором распыла

## Версия и сборка

`build_id` — SHA-256 исходников. Версии Python, NumPy, Pillow, zlib, libtiff, Tcl/Tk записываются в `config.json` серии.

---

**Сборка:** `0.1.0` | **Профиль:** `standard-v1` | **Python:** 3.14 x64