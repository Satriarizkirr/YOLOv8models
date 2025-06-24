import os
import uuid
import time
import threading
from datetime import datetime
from dotenv import load_dotenv
import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory, url_for
from flask_cors import CORS
from flask_socketio import SocketIO
from ultralytics import YOLO

from project.config import Detection, create_session

# Muat variabel dari file .env
load_dotenv()

# 1. INISIALISASI & KONFIGURASI
app = Flask(__name__, static_folder='static')

# --- Konfigurasi dari .env ---
HOST = os.getenv('FLASK_HOST', 'localhost')
PORT = int(os.getenv('FLASK_PORT', 5000))
DEBUG_MODE = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
CORS_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', '*')

app.config['SERVER_NAME'] = f"{HOST}:{PORT}"

CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})
socketio = SocketIO(app, cors_allowed_origins=CORS_ORIGINS, async_mode='threading')

# 2. FUNGSI HELPER UNTUK LOGGING REAL-TIME
def send_log(message, log_type='INFO'):
    """
    Mencetak log ke konsol dan mengirimkannya ke klien melalui Socket.IO.
    log_type bisa: INFO, DEBUG, SUCCESS, WARNING, ERROR, CMD
    """
    timestamp = datetime.now().strftime('%H:%M:%S')
    # Tetap cetak di terminal untuk debugging server
    print(f"[{timestamp}] [{log_type}] {message}")
    # Kirim ke frontend
    socketio.emit('log_message', {'time': timestamp, 'message': message, 'type': log_type})

# 3. KONFIGURASI MODEL & VARIABEL GLOBAL
MODEL_PATH = os.getenv('MODEL_PATH')
MODEL_FILE = os.getenv('YOLO_MODEL_FILE')
YOLO_CONF = float(os.getenv('YOLO_CONF_THRESHOLD', 0.40))
YOLO_IOU = float(os.getenv('YOLO_IOU_THRESHOLD', 0.40))
model = YOLO(os.path.join(MODEL_PATH, MODEL_FILE))

detection_dir = os.path.join(app.static_folder)
os.makedirs(detection_dir, exist_ok=True)

ESP8266_FORWARD_URL = os.getenv('ESP8266_FORWARD_URL')
ESP8266_STOP_URL = os.getenv('ESP8266_STOP_URL')
ESP32CAM_CAPTURE_URL = os.getenv('ESP32CAM_CAPTURE_URL')

app_state = {'is_running': False, 'process_thread': None}

# 4. FUNGSI INTI & PROSES
def process_yolo_and_broadcast(image_frame):
    try:
        results = model(image_frame, conf=YOLO_CONF, iou=YOLO_IOU)
        r = results[0] if results and len(results) > 0 else None
        
        if not r or len(r.boxes) == 0:
            send_log("Tidak ada objek terdeteksi pada gambar.", log_type='DEBUG')
            return

        annotated_img = r.plot()
        annot_filename = f"detection_{uuid.uuid4()}.jpg"
        annot_path = os.path.join(detection_dir, annot_filename)
        cv2.imwrite(annot_path, annotated_img)
        
        session = create_session()
        try:
            now = datetime.now()
            for box in r.boxes:
                # ... (logika deteksi tetap sama) ...
                class_name_detected = model.names[int(box.cls[0])]
                conf_score = float(box.conf[0])
                new_detection_record = Detection(
                    detected_at=now, class_=class_name_detected, confidence=round(conf_score, 4),
                    image_path=annot_filename, created_at=now, updated_at=now
                )
                session.add(new_detection_record)
                session.flush()
                
                image_url = url_for('static', filename=annot_filename, _external=True)
                socket_data = {
                    'id': new_detection_record.id, 'detected_at': now.strftime('%Y-%m-%d %H:%M:%S'),
                    'class_': class_name_detected, 'className': class_name_detected.lower(),
                    'confidence': round(conf_score, 4), 'image_url': image_url
                }
                socketio.emit('new_detection', socket_data)
            
            session.commit()
            send_log(f"✅ Deteksi berhasil! {len(r.boxes)} objek disimpan ke DB.", log_type='SUCCESS')
        
        except Exception as e:
            session.rollback()
            send_log(f"Error saat menyimpan ke DB: {e}", log_type='ERROR')
        finally:
            session.close()
            
    except Exception as e:
        send_log(f"Error fatal di dalam fungsi proses YOLO: {e}", log_type='ERROR')

def autonomous_loop():
    """Loop ini berjalan di background, mengontrol alat secara otomatis."""
    with app.app_context():
        send_log("🤖 Loop otomatis DIMULAI.", log_type='INFO')
        while app_state.get('is_running'):
            send_log("--- Memulai siklus baru ---", log_type='INFO')
            try:
                if not ESP8266_STOP_URL:
                    send_log("ESP8266_STOP_URL tidak diatur di .env", log_type='ERROR')
                    time.sleep(10); continue

                requests.get(ESP8266_STOP_URL, timeout=5)
                send_log("Mengirim perintah STOP.", log_type='CMD')
                time.sleep(5)
                
                if not app_state.get('is_running'): break
                
                if not ESP32CAM_CAPTURE_URL:
                    send_log("ESP32CAM_CAPTURE_URL tidak diatur di .env", log_type='ERROR')
                    time.sleep(10); continue
                
                send_log("Mengambil gambar dari kamera...", log_type='DEBUG')
                response = requests.get(ESP32CAM_CAPTURE_URL, timeout=10)
                
                if response.status_code == 200:
                    send_log("Gambar diterima, memproses...", log_type='DEBUG')
                    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        process_yolo_and_broadcast(frame)
                    else:
                        send_log("Gagal decode gambar dari kamera.", log_type='WARNING')
                else:
                    send_log(f"Kamera tidak merespon (Status: {response.status_code})", log_type='WARNING')
                
                if not app_state.get('is_running'): break

                if not ESP8266_FORWARD_URL:
                    send_log("ESP8266_FORWARD_URL tidak diatur di .env", log_type='ERROR')
                    time.sleep(10); continue

                requests.get(ESP8266_FORWARD_URL, timeout=0.5)
                send_log("Mengirim perintah START (maju).", log_type='CMD')
                time.sleep(0.5)
            except requests.exceptions.RequestException as e:
                send_log(f"Error Koneksi: Tidak bisa terhubung ke alat. {e}", log_type='ERROR')
                time.sleep(0.5)
            except Exception as e:
                send_log(f"Error tak terduga dalam siklus: {e}", log_type='ERROR')
                time.sleep(5)
    send_log("🛑 Loop otomatis DIHENTIKAN.", log_type='INFO')

# 5. ENDPOINT API (DENGAN LOGGING)
@app.route('/api/start-process', methods=['POST'])
def start_process():
    if not app_state.get('is_running'):
        app_state['is_running'] = True
        if not all([ESP8266_FORWARD_URL, ESP8266_STOP_URL, ESP32CAM_CAPTURE_URL]):
            app_state['is_running'] = False
            msg = "Konfigurasi URL untuk ESP tidak lengkap di file .env."
            send_log(msg, log_type='ERROR')
            return jsonify({"error": msg}), 500
        app_state['process_thread'] = threading.Thread(target=autonomous_loop)
        app_state['process_thread'].start()
        send_log("Proses berhasil dimulai oleh pengguna.", log_type='INFO')
        return jsonify({"message": "Proses otomatis berhasil dimulai."}), 200
    return jsonify({"message": "Proses sudah berjalan dari sebelumnya."}), 400

@app.route('/api/stop-process', methods=['POST'])
def stop_process():
    if app_state.get('is_running'):
        app_state['is_running'] = False
        send_log("Perintah berhenti diterima. Proses akan berhenti setelah siklus ini selesai.", log_type='INFO')
        return jsonify({"message": "Perintah berhenti dikirim. Proses akan berhenti setelah siklus ini selesai."}), 200
    return jsonify({"message": "Proses tidak sedang berjalan."}), 400

@app.route('/api/process-status', methods=['GET'])
def process_status():
    return jsonify({"is_running": app_state.get('is_running', False)})

@app.route('/api/detect', methods=['POST'])
def detect_damage_from_upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No image provided'}), 400
    try:
        send_log("Menerima gambar via upload...", log_type='INFO')
        img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        process_yolo_and_broadcast(frame)
        return jsonify({'success': True, 'message': 'Gambar berhasil diproses.'}), 200
    except Exception as e:
        send_log(f"Gagal memproses gambar yang diupload: {str(e)}", log_type='ERROR')
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
        send_log(f"Seluruh ({num_deleted}) data deteksi telah dihapus oleh pengguna.", log_type='WARNING')
        socketio.emit('database_cleared')
        return jsonify({'message': f'Berhasil menghapus {num_deleted} data.'}), 200
    except Exception as e:
        session.rollback()
        send_log(f"Gagal menghapus data: {str(e)}", log_type='ERROR')
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

# 6. MENJALANKAN APLIKASI
if __name__ == '__main__':
    send_log(f"Server dimulai pada http://{HOST}:{PORT}", log_type='INFO')
    socketio.run(app, host=HOST, port=PORT, allow_unsafe_werkzeug=True, debug=DEBUG_MODE)