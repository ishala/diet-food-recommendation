import sys
import os

# Tambahkan jalur aplikasi ke sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/app")

from app import app as application
