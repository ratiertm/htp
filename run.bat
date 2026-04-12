@echo off
echo ========================================
echo   Hub Topology Runtime 3D Dashboard
echo ========================================
echo   URL: http://localhost:8765
echo ========================================

cd /d "%~dp0"
call C:\Users\Mindbuild\anaconda3\Scripts\activate.bat skild
python server.py
pause
