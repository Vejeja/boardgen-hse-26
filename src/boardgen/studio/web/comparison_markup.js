export const comparisonMarkup = `
  <section class="selection" aria-label="Выбор для YOLO">
    <h2>Выбрано для YOLO: <span id="selectionCount">…</span></h2>
    <ul id="selectionList" style="max-height:180px;overflow:auto"></ul>
    <p id="selectionHint">Загрузка выбора проекта…</p>
    <div class="toolbar"><button id="export" disabled>Скачать мой выбор</button></div>
    <p>Выбор определяет входы следующего рендера. Генерация и обучение запускаются отдельно.</p>
  </section>
  <section class="selection" aria-label="Кандидаты для сравнения">
    <div class="selection-heading"><h2>Кандидаты</h2><span id="inventory"></span></div>
    <p id="availability"></p>
    <label><input id="showDiagnostics" type="checkbox"> Показать сохранённые варианты с известными дефектами</label>
    <div id="candidates"></div><div id="separate-candidates"></div>
  </section>
  <div class="toolbar controls">
    <button id="geometry" aria-pressed="true">Серая геометрия</button><button id="material" aria-pressed="false">Материал</button><button id="silhouette" aria-pressed="false">Силуэт</button>
    <label><input id="overlay" type="checkbox" checked> Контур маски на моделях</label>
    <button id="free" aria-pressed="false">Общее вращение</button><button id="restore">Ракурс маски</button>
  </div>
  <div class="comparison-layout">
    <section id="modelArea" aria-label="3D-кандидаты"><div id="modelGrid"></div></section>
    <aside class="reference" aria-label="Исходная маска и поза">
      <h2>Исходная маска</h2>
      <label class="frame-label">Поза из видео <select id="frame"></select></label>
      <div class="frame-navigation"><button id="previous" aria-label="Предыдущая поза">←</button><input id="frameSlider" type="range" min="0" max="0" value="0" step="1" aria-label="Номер позы"><button id="next" aria-label="Следующая поза">→</button></div>
      <p id="frameInfo"></p>
      <div class="reference-modes"><button id="showMask" aria-pressed="true">Маска</button><button id="showPhoto" aria-pressed="false">Исходное фото</button></div>
      <img id="referenceImage" alt="Маска выбранного кадра">
      <p id="alignment"></p>
      <p class="hint">Камеры получены при реконструкции. Маска может ошибочно заполнять промежутки — сверяйте их с фото. Это визуальное сравнение, а не независимая оценка точности.</p>
      <p id="favorite"></p>
    </aside>
  </div>
  <p id="status" role="status">Загрузка…</p>
`;
