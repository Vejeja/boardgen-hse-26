# Настройки и запуск

Основной порядок работы описан в [README](../README.md).

## Проект и данные

`PROJECT` — папка проекта. При первом открытии программа создаёт `project.yaml`
с настройками и `.boardgen/` с заданиями и результатами. Относительные пути
считаются от этой папки; внешние папки нужно подключить через интерфейс.

[Пример настроек](project.example.yaml) использует данные репозитория.
Его можно сохранить как локальный `project.yaml` в корне репозитория.
Для проекта в другой папке замените пути на абсолютные и подключите папку с данными.

## Настройки

- `figures` — названия фигур, видео, границы фрагмента, проверенные кадры и камеры.
- `prepare` — маски, форма и материал; общие параметры и настройки отдельных фигур.
- `scenes` — выбранные модели, классы, фоны, камеры, свет и расположение фигур.
- `dataset` — разделение выборок, дубликаты, фоны и реальные размеченные изображения.
- `training` — начальные веса YOLO, размер изображения и параметры обучения.
- `evaluation` — выборка и пороги распознавания.

Формат `boardgen.studio.v1` описан в [JSON Schema](project.schema.json).
Неизвестные поля и несовместимые настройки программа отклоняет до запуска.
Изменения действуют со следующего запуска и не заменяют выбранную модель:
её нужно выбрать заново в «Подготовка → Сравнение и выбор».

MV строит форму по четырём проверенным сторонам и поддерживает фототекстуру
или нейтральный материал. Построение через карты глубины доступно для Spirula.

## Окружение

Нужны macOS arm64 с Metal, Python 3.13 и FFmpeg в `PATH`.
`make setup-full` устанавливает зависимости из `uv.lock`, `make setup-tools` —
Blender 4.5.14, Spirula 2026.9.30, OpenMVS 2.4.0, Hunyuan3D-Swift и веса Hunyuan3D-2mv.
Установщик проверяет SHA256 и версию исходников, использует готовые загрузки повторно
и сохраняет отчёт в `out/setup/external-tools.json`.
BiRefNet загружается через кэш Hugging Face без обучения на фигурах проекта.

Переменные окружения:

- `BOARDGEN_HOME` — корень репозитория, вне него — `~/.cache/boardgen`;
  здесь находятся общие данные и файл блокировки вычислений.
- `BOARDGEN_TOOLS_DIR` — внешние программы, по умолчанию `$BOARDGEN_HOME/tmp/benchmark_tools`.
- `BLENDER_BIN` — путь к Blender. Без него поиск идёт по локальной установке,
  `PATH`, затем `/Applications/Blender.app/Contents/MacOS/Blender`.
- `SPIRULA_BIN` — путь к Spirula, по умолчанию
  `$BOARDGEN_TOOLS_DIR/Spirula Studio.app/Contents/MacOS/spirula`.
- `STUDIO_PORT` — порт `make studio`, по умолчанию `8769`.

```bash
make doctor PROJECT=/absolute/path/to/project
make studio PROJECT=/absolute/path/to/project STUDIO_PORT=8769
```

`doctor` проверяет наличие зависимостей и внешних программ, но не качество моделей
или работу GPU под нагрузкой. `make setup` ставит базовые зависимости,
`make setup-neural` добавляет нейросегментацию, `make setup-full` — все методы.

## Команды

Терминал и веб-интерфейс используют общие настройки.

```bash
.venv/bin/boardgen-project --help
.venv/bin/boardgen-project --project /absolute/path/to/project validate
.venv/bin/boardgen-project --project /absolute/path/to/project list
.venv/bin/boardgen-project schema
```

`validate` проверяет настройки, `list` показывает задания и версии, `schema` выводит схему.
Команда `run` запускает этап: `frames`, `prepare`, `prepare-candidates`, `pilot`,
`dataset`, `train`, `evaluate`, `predict`, `localize` или `install`.
Параметры передаются JSON-файлом через `--parameters`; допустимые поля описаны
в `src/boardgen/studio/parameters.py`. Открытие страницы не запускает вычисления.

Для проверки выбранной модели и генератора сцен:

```bash
make asset-smoke MESH=/absolute/path/to/selected.glb
make pipeline-smoke MESH=/absolute/path/to/selected.glb
```

Эти команды требуют внешних программ. Обычные тесты и `make studio-check`
не скачивают веса и не запускают обучение.
