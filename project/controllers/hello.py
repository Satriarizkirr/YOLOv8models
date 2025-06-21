import os
import uuid
import time
import threading
from datetime import datetime

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory, url_for
from flask_cors import CORS
from flask_socketio import SocketIO
from ultralytics import YOLO

from project.config import Detection, create_session

# ==============================================================================
# 1. INISIALISASI & KONFIGURASI
# ==============================================================================
app = Flask(__name__, static_folder='static')
# ==============================================================================
# --- INI DIA PERBAIKANNYA ---
app.config['SERVER_NAME'] = 'localhost:5000'
# ==============================================================================
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ... (Semua konfigurasi Model, Path, URL, dan app_state tetap sama) ...
MODEL_PATH = r'C:\Users\DELL\Documents\PROPOSAL SKRIPSI\YOLOv8model\project\controllers'
model = YOLO(os.path.join(MODEL_PATH, 'bestv8m120.pt'))
detection_dir = os.path.join(app.static_folder)
os.makedirs(detection_dir, exist_ok=True)
ESP8266_FORWARD_URL = 'http://192.168.194.202/start'
ESP8266_STOP_URL = 'http://192.168.194.202/stop'
ESP32CAM_CAPTURE_URL = 'http://192.168.194.5/capture'
app_state = {'is_running': False, 'process_thread': None}

# ==============================================================================
# 2. FUNGSI INTI (Tidak ada perubahan)
# ==============================================================================
def process_yolo_and_broadcast(image_frame):
    try:
        results = model(image_frame, conf=0.40, iou=0.40)
        r = results[0] if results and len(results) > 0 else None
        
        if not r or len(r.boxes) == 0:
            print("INFO: Tidak ada objek terdeteksi pada gambar.")
            return

        annotated_img = r.plot()
        annot_filename = f"detection_{uuid.uuid4()}.jpg"
        annot_path = os.path.join(detection_dir, annot_filename)
        cv2.imwrite(annot_path, annotated_img)
        
        session = create_session()
        try:
            now = datetime.now()
            for box in r.boxes:
                class_name_detected = model.names[int(box.cls[0])]
                conf_score = float(box.conf[0])
                
                new_detection_record = Detection(
                    detected_at=now, class_=class_name_detected, confidence=round(conf_score, 4),
                    image_path=annot_filename, created_at=now, updated_at=now
                )
                session.add(new_detection_record)
                session.flush()
                
                # Sekarang pemanggilan url_for ini aman karena SERVER_NAME sudah di-set
                image_url = url_for('static', filename=annot_filename, _external=True)
                socket_data = {
                    'id': new_detection_record.id, 'detected_at': now.strftime('%Y-%m-%d %H:%M:%S'),
                    'class_': class_name_detected, 'className': class_name_detected.lower(),
                    'confidence': round(conf_score, 4), 'image_url': image_url
                }
                socketio.emit('new_detection', socket_data)
            
            session.commit()
            print(f"✅ Deteksi berhasil disimpan ke DB ({len(r.boxes)} objek).")
        
        except Exception as e:
            session.rollback()
            print(f"❌ Error saat menyimpan ke DB: {e}")
        finally:
            session.close()
            
    except Exception as e:
        print(f"❌ Error fatal di dalam fungsi proses YOLO: {e}")


# 3. LOGIKA PROSES OTOMATIS (DENGAN LOG DEBUG)

def autonomous_loop():
    """Loop ini berjalan di background, mengontrol alat secara otomatis."""
    with app.app_context():
        print("🤖 Loop otomatis DIMULAI di background...")
        while app_state.get('is_running'):
            print("\n" + "="*20 + " SIKLUS BARU " + "="*20)
            try:
                requests.get(ESP8266_STOP_URL, timeout=5)
                print("CMD: STOP terkirim.")
                time.sleep(5)
                
                if not app_state.get('is_running'): break
                
                print(">>> [DEBUG] Mencoba mengambil gambar dari kamera di URL:", ESP32CAM_CAPTURE_URL)
                response = requests.get(ESP32CAM_CAPTURE_URL, timeout=10)
                print(f">>> [DEBUG] Kamera merespon dengan status: {response.status_code}")
                
                if response.status_code == 200:
                    print(">>> [DEBUG] Mencoba decode gambar...")
                    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        print(">>> [DEBUG] Gambar berhasil di-decode, memanggil proses YOLO...")
                        process_yolo_and_broadcast(frame)
                        print(">>> [DEBUG] Panggilan ke proses YOLO selesai.")
                    else:
                        print("⚠️ Gagal decode gambar dari kamera.")
                else:
                    print(f"⚠️ Kamera tidak merespon dengan baik (Status: {response.status_code})")
                
                if not app_state.get('is_running'): break

                requests.get(ESP8266_FORWARD_URL, timeout=5)
                print("CMD: START (maju) terkirim.")
                time.sleep(5)
            except requests.exceptions.RequestException as e:
                print(f"❌ TERJADI ERROR KONEKSI: Tidak bisa terhubung ke alat. Pastikan IP dan jaringan benar. Detail: {e}")
                time.sleep(5)
            except Exception as e:
                print(f"❌ Terjadi error tak terduga dalam satu siklus: {e}")
                time.sleep(5)
    print("🛑 Loop otomatis DIHENTIKAN.")


# 4. ENDPOINT API (PINTU UNTUK VUE)


# API UNTUK KONTROL PROSES
@app.route('/api/start-process', methods=['POST'])
def start_process():
    if not app_state.get('is_running'):
        app_state['is_running'] = True
        app_state['process_thread'] = threading.Thread(target=autonomous_loop)
        app_state['process_thread'].start()
        return jsonify({"message": "Proses otomatis berhasil dimulai."}), 200
    return jsonify({"message": "Proses sudah berjalan dari sebelumnya."}), 400

@app.route('/api/stop-process', methods=['POST'])
def stop_process():
    if app_state.get('is_running'):
        app_state['is_running'] = False
        return jsonify({"message": "Perintah berhenti dikirim. Proses akan berhenti setelah siklus ini selesai."}), 200
    return jsonify({"message": "Proses tidak sedang berjalan."}), 400

@app.route('/api/process-status', methods=['GET'])
def process_status():
    return jsonify({"is_running": app_state.get('is_running', False)})

# API UNTUK DATA & FILE
@app.route('/api/detect', methods=['POST'])
def detect_damage_from_upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No image provided'}), 400
    try:
        img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        process_yolo_and_broadcast(frame)
        return jsonify({'success': True, 'message': 'Gambar berhasil diproses.'}), 200
    except Exception as e:
        return jsonify({'error': f'Gagal memproses gambar yang diupload: {str(e)}'}), 500

@app.route('/api/detections-history', methods=['GET'])
def get_detections_history():
    session = create_session()
    try:
        detections_from_db = session.query(Detection).order_by(Detection.detected_at.desc()).all()
        result = [
            {
                'id': det.id, 'class_': det.class_, 'className': det.class_.lower(),
                'confidence': round(det.confidence, 4), 'detected_at': det.detected_at.strftime('%Y-%m-%d %H:%M:%S'),
                'image_url': url_for('static', filename=det.image_path, _external=True)
            } for det in detections_from_db
        ]
        return jsonify(result), 200
    finally:
        session.close()

@app.route('/api/delete/all', methods=['DELETE'])
def delete_all_detections():
    session = create_session()
    try:
        num_deleted = session.query(Detection).delete()
        session.commit()
        socketio.emit('database_cleared')
        return jsonify({'message': f'Berhasil menghapus {num_deleted} data.'}), 200
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

# ==============================================================================
# 5. MENJALANKAN APLIKASI
# ==============================================================================
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True, debug=True)