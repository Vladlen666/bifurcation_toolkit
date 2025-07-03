# Dynamic System Sandbox

Интерактивная песочница для анализа двух-параметрических систем дифференциальных уравнений.

**Функционал**  
- Редактор системы (SymPy-синтаксис)  
- Двойной клик для построения изоклин и фазовых траекторий  
- Настройка диапазона параметров (μ₁, μ₂) и фазовых осей (x, y)  
- Выбор метода интеграции (RK45, RK23, DOP853, LSODA)  
- История траекторий с кнопкой «Clear»  
- Интерактивный zoom/pan (NavigationToolbar2QT + колесо мыши)  
- Кнопки «Max Param» / «Max Phase» для развёртывания одного холста

## Установка и запуск

```bash
git clone https://github.com/<ваш-логин>/dynamic-sandbox.git
cd dynamic-sandbox
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt

python pyqt_dynamic_system_demo.py
