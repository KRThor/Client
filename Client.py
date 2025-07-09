from multiprocessing import Queue as MQueue
from multiprocessing import Process
from datetime import datetime
from tabulate import tabulate
from queue import Queue
import numpy as np
import subprocess
import py_compile
import threading
import traceback
import pickle
import socket
import shutil
import stat
import json
import time
import git
import cv2
import os

from lib.log import LogManager
from lib.galaxycamera import GalaxyCamera
from lib.classification import Classification
from lib.detection import Detection

class SocketServer:
    # HOST = "192.168.0.100"
    # PORT = 9999
    HOST = "192.168.50.9"
    PORT = 9999
    
    def __init__(self):
        self.host = self.HOST
        self.port = self.PORT
        self.client_socket = None
        self.socket_connected = False
        self.load_complete = False  # 모든 마스킹, 모델 로드 완료 여부
        self.send_queue = Queue()

        # 소켓 통신을 통해 변하는 변수
        self.info_json = None
        self.model_name = None
        self.current_date = None
        self.capture_signal_dict = {"origin": False, "ng": False, "inspection": False}
        self.recent_origin_send_signal = True  # 최근 원본 이미지 전송 여부

    def connect_to_server(self):
        """메인 서버 연결"""
        while not self.socket_connected:
            try:
                if self.client_socket:
                    self.client_socket.close()

                LM.log_print("[Socket] Attempting to connect to the server...")
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.host, self.port))
                self.socket_connected = True
                LM.log_print("[Socket] Successfully connected to the server")
            except:
                LM.log_print(f"[Socket] Failed to connect to the server : {traceback.format_exc()}")
                time.sleep(1)

    def send_thread(self): 
        """데이터 전송 스레드"""
        while True:
            if not self.send_queue.empty():
                type, data = self.send_queue.get()
                try:
                    if type == 'msg':
                        data = data.encode() if isinstance(data, str) else data
                        data_length = len(data)
                        self.client_socket.sendall(data_length.to_bytes(4, byteorder='big'))
                        self.client_socket.sendall(data)
                        print(f"[Socket] {type} send complete!: {data}")
                    elif type == 'image':
                        image_pickle = pickle.dumps(data)
                        length_bytes = len(image_pickle).to_bytes(4, byteorder='big')
                        self.client_socket.sendall(length_bytes)
                        self.client_socket.sendall(image_pickle)
                        print(f"[Socket] image send complete!")
                except :
                    LM.log_print(f"[Socket] {type} send error: {traceback.format_exc()}")
                    try:
                        self.client_socket.close()
                        self.socket_connected = False
                    except:
                        pass
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def recvall(self, sock, count):
        buf = b""
        while count:
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def git_clone(self, git_url):
        """깃허브 코드 다운로드"""
        try:
            target_dir = os.getcwd()  # 현재 경로
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            git.Git(target_dir).clone(git_url)
        except:
            LM.log_print(f"[Git] Failed to clone repository: {traceback.format_exc()}")
        
    def load_info_json(self):
        """모델 정보 로드"""
        try:
            self.detection_use = self.info_json['detection_use']  # 각인 디텍션 검사 실시 여부
            self.detection_frame = self.info_json['detection_frame']  # 각인 디텍션 검사 프레임
            self.capture_signal_dict['origin'] = self.info_json['origin_image_capture']  # 원본 이미지 캡처 여부
            self.capture_signal_dict['ng'] = self.info_json['ng_image_capture']  # 불량 이미지 캡처 여부
            self.capture_signal_dict['inspection'] = self.info_json['inspection_image_capture']  # 검사 이미지 캡처 여부
            self.show_coord = self.info_json['show_coord']  # 화면 출력용 좌표
            self.detection_coords = self.info_json['detection_coords']  # 각인 디텍션 검사 좌표
            self.ok_engrave = self.info_json['ok_engrave']  # 정상 각인 개수
            self.inspection_frame = self.info_json['inspection_frame']  # 검사 프레임 수
            self.inspection_coords = self.info_json['inspection_coords']  # 검사 좌표
            self.critical_ng_list = self.info_json['critical_ng_list']  # 중대 불량 목록
            self.label_ng_conditions = self.info_json['label_ng_conditions']  # 불량 판정 조건
            self.json_label_list = self.info_json['label_ng_conditions'].keys()  # 불량 라벨 목록
            LM.log_print(f"[Model] info_json loaded successfully")
        except:
            LM.log_print(f"[Model] Failed to load info_json: {traceback.format_exc()}")

    def load_result_dict(self):
        """검사 결과 딕셔너리 로드"""
        HW.inspection_result_dict = {}
        for part_name, values in self.inspection_coords.items():
            HW.inspection_result_dict[part_name] = []
        print(f"[LOADED] result_dict loaded successfully")

    def load_classification_mask(self):
        """분류 마스크 로드"""
        try:
            # 검사 영역 및 마스크 로드
            classi_mask_path_list = []
            for part_name, values in self.inspection_coords.items():
                classi_mask_path_list.append(f'mask/classification/{self.model_name}{values[0]}.png')
            classi_mask_path_list = list(set(classi_mask_path_list))

            # 분류 마스크 로드
            if classi_mask_path_list:
                result = CL.load_mask(classi_mask_path_list)  # result: fail or {'status': 'success', 'main': image, 'sub': image}
            
                if result == 'fail':
                    self.send_queue.put(('msg', 'load:classi_masking:fail'))
                    self.load_complete = False
                    raise Exception("Classification mask loading failed")
                else:
                    HW.mask_images['main'] = result.get('main')
                    HW.mask_images['sub'] = result.get('sub')
                    LM.log_print(f"[MASK] {self.model_name} Classification Mask Loaded")
            else:
                LM.log_print(f"[MASK] {self.model_name} Classification Mask path is empty")
        except Exception as e:
            raise

    def load_detection_mask(self):
        """디텍션 마스크 로드"""
        try:
            det_mask_path = f'mask/detection/{self.model_name}.png'
            result = DT.load_mask(det_mask_path)  # result: ('fail or success', 'mask_image')
            
            if isinstance(result, str) and result == 'fail':
                self.send_queue.put(('msg', 'load:det_masking:fail'))
                self.load_complete = False
                raise Exception("Detection mask loading failed")
            else:
                HW.mask_images['det'] = result
                LM.log_print(f"[MASK] {self.model_name} Detection Mask Loaded")
        except Exception as e:
            raise
    
    def run(self):
        while True:
            time.sleep(0.01)
            if not self.socket_connected:
                self.connect_to_server()
                
            try:
                header = self.recvall(self.client_socket, 4)
                if not header:
                    LM.log_print("[Socket] Header not received")
                    self.socket_connected = False
                    continue

                data_length = int.from_bytes(header, 'big')
                data = self.recvall(self.client_socket, data_length)
                if not data:
                    LM.log_print("[Socket] Data not received")
                    self.socket_connected = False
                    continue
            
                try:
                    recv_data = data.decode('utf-8').strip()
                    # LM.log_print(f"[Socket] received data: {recv_data}")

                    if "json" in recv_data:
                        json_data = recv_data.split(":", 1)[1].strip()
                        self.info_json = json.loads(json_data)
                        print(f"[information] info_json: {self.info_json}")
                        self.load_info_json()  # Json값들 변수에 저장
                        self.load_result_dict()  # 검사 결과 딕셔너리 로드
                        self.load_classification_mask()  # 마스킹 로드
                    
                    elif "model" in recv_data:
                        recv_model_name = recv_data.split(":", 2)[1].strip()
                        recv_json_data = recv_data.split(":", 2)[2].strip()

                        if self.load_complete and self.model_name == recv_model_name:  # 모델 로드 완료 후 같은 모델 로드 요청 시 패스
                            self.send_queue.put(('msg', 'model_load_complete'))
                            LM.log_print(f"[Socket] {self.model_name} Loading Signal Recv, SameModel Setup (pass)")
                            continue

                        else:
                            self.model_name = recv_model_name
                            self.info_json = json.loads(recv_json_data)
                            LM.log_print(f"[Socket] {self.model_name} Loading Signal Recv, New Model Setup")
                            print(f"[information] info_json: {self.info_json}")
                            self.load_info_json()  # Json값들 변수에 저장

                            try:
                                # 1. 기존 monitor 폴더 제거
                                if os.path.isdir('monitor'):
                                    shutil.rmtree('monitor')

                                # 2. 분류 모델
                                self.load_result_dict()  # 검사 결과 딕셔너리 로드
                                self.load_classification_mask()  # 마스크 로드

                                # 분류 모델 로드
                                result, update_msg = CL.model_load(self.model_name, self.json_label_list)
                                if result == 'fail':
                                    self.send_queue.put(('msg', 'load:classi_model:fail'))
                                    self.load_complete = False
                                    raise Exception("Classification model loading failed")
                                elif result == 'different':
                                    self.send_queue.put(('msg', update_msg))
                                else:
                                    LM.log_print(f"[MODEL] {self.model_name} Classification Model Loaded")

                                # 3. 디텍션
                                self.load_detection_mask()  # 마스크 로드

                                # 디텍션 모델 로드
                                result = DT.model_load()
                                if result == 'fail':
                                    self.send_queue.put(('msg', 'load:det_model:fail'))
                                    self.load_complete = False
                                    raise Exception("Detection model loading failed")
                                else:
                                    print(f"[MODEL] {self.model_name} Detection Model Loaded")

                                # 4. 모든 로딩 성공 완료 신호 전송
                                self.send_queue.put(('msg', 'model_load_complete'))
                                LM.log_print(f"[Model] {self.model_name} All models loaded successfully!!")
                                self.load_complete = True

                            except Exception as e:
                                self.load_complete = False
                                LM.log_print(f"[Model] {self.model_name} Failed to load models: {str(e)}")

                    elif recv_data == 'start':  # 검사 시작
                        LM.log_print("[Inspection] Start Signal Recv")
                        HW.reset_inspection_variables()
                        HW.state = 'RUNNING'
                        HW.inspection_start_time = time.time()
                        GC.shooting_signal = True
                        self.current_date = datetime.now().strftime("%Y_%m_%d")
                    
                    elif "capture" in recv_data:
                        capture_type = recv_data.split(":")[1]  # origin(원본), ng(불량), inspection(검사이미지)
                        capture_mode_str = recv_data.split(":")[2].lower()  # true, false
                        capture_mode = True if capture_mode_str == "true" else False
                        self.capture_signal_dict[capture_type] = capture_mode
                        LM.log_print(f"[Socket] Capture: {capture_type} {capture_mode}")
                    
                    elif "reboot" in recv_data:
                        LM.log_print("[Socket] Reboot Signal Recv")
                        os.system("sudo reboot")
                    
                    elif "recent" in recv_data:
                        recent_origin_send_signal = recv_data.split(":")[2]  # start, stop
                        if recent_origin_send_signal == 'start':
                            self.recent_origin_send_signal = True
                            LM.log_print("[Socket] Recent Origin Send Signal Start")
                        elif recent_origin_send_signal == 'stop':
                            self.recent_origin_send_signal = False
                            LM.log_print("[Socket] Recent Origin Send Signal Stop")
                    
                    elif recv_data == 'code_update':
                        LM.log_print("[Socket] Code Update Signal Recv")
                        HW.code_update()

                except:
                    LM.log_print(f"[Socket] Socket command execution failed: {traceback.format_exc()}")
                    time.sleep(1)
                    raise
            except:
                LM.log_print(f"[Socket] Socket Error: {traceback.format_exc()}")
                time.sleep(1)

class HardWork:
    def __init__(self):
        # 마스킹 이미지들을 딕셔너리로 관리
        self.mask_images = {
            'main': None,  # 일반 마스킹 이미지
            'sub': None,   # 서브 마스킹 이미지
            'det': None    # 디텍션 마스킹 이미지   
        }
        self.trash_frame_count = 0  # 버리는 프레임 체크용
        self.inspection_count = 0  # 검사 프레임 수
        self.inspection_start_time = None  # 검사 시간
        self.skip_unnecessary_images = False  # 불필요한 이미지 스킵 여부
        self.first_frame_check = False  # 첫 프레임 체크
        self.ok_image = None  # 양품 이미지
        self.det_ng_image = None  # 디텍션 불량 이미지
        self.ng_name = None  # 불량 이름
        self.detection_ng_count = 0  # 디텍션 검사 불량 개수
        self.inspection_result_dict = {}  # 검사 결과 목록
        self.image_save_queue = MQueue()  # 이미지 저장 큐
        self.image_monitor_queue = MQueue()  # 이미지 모니터링 큐
        self.origin_image_save_base_path = 'Capture/origin'  # 원본 이미지 저장 경로
        self.ng_image_save_base_path = 'Capture/ng'  # 불량 이미지 저장 경로
        self.classification_image_save_base_path = 'Capture/inspection/classification'  # 분류 검사 이미지 저장 경로
        self.detection_image_save_base_path = 'Capture/inspection/detection'  # 디텍션 검사 이미지 저장 경로
        self.state = 'IDLE'  # IDLE(초기), RUNNING(검사중), ANALYZING(분석중), WAITING(대기)  # 클라이언트 상태
        self.lock = threading.Lock()
        self.recent_image_send_count = {}  # 최근 이미지 전송 횟수

    def reset_inspection_variables(self):
        """검사 변수 초기화"""
        with self.lock:
            GC.Qimage.queue.clear()
            self.trash_frame_count = 0
            self.inspection_count = 0
            self.recent_image_send_count = {}
            self.skip_unnecessary_images = False
            self.first_frame_check = False
            self.ok_image = None
            self.det_ng_image = None
            self.ng_name = None
            self.detection_ng_count = 0
            self.recent_image_send_count['ORIGIN'] = 0
            self.recent_image_send_count['ENGRAVE'] = {'OK': 0, 'NG': 0}
            for part_name in self.inspection_result_dict.keys():
                self.inspection_result_dict[part_name] = []
                self.recent_image_send_count[part_name] = {'OK': 0, 'NG': 0}
            self.state = 'IDLE'
            LM.log_print("[Inspection] Inspection Variables Reset")
    
    def inspection_thread(self):
        while True:
            try:
                if self.state == 'IDLE':
                    time.sleep(0.1)
                    continue

                elif self.state == 'RUNNING':
                    if not GC.Qimage.empty():
                        if self.inspection_count >= SS.inspection_frame:
                            LM.log_print(f"[Inspection] Inspection Frame Over: {self.inspection_count} >= {SS.inspection_frame}")
                            GC.shooting_signal = False
                            self.state = 'ANALYZING'
                            continue
                        
                        image = GC.Qimage.get()
                        
                        if not self.skip_unnecessary_images:  # 첫 4프레임 스킵
                            self.trash_frame_count += 1
                            if self.trash_frame_count == 4:
                                LM.log_print(f"[Inspection] Skip Unnecessary Images: {self.trash_frame_count}")
                                self.skip_unnecessary_images = True
                                self.inspection_count = 0
                            else:
                                continue
                        
                        self.inspection_count += 1

                        print(f"Inspection frame: {self.inspection_count}")

                        if SS.capture_signal_dict["origin"]:  # 원본 이미지 저장
                            origin_image_save_path = f"{self.origin_image_save_base_path}/{SS.current_date}/{SS.model_name}"
                            self.image_save_queue.put((origin_image_save_path, image))

                        if not self.first_frame_check:  # 첫 프레임 양품 이미지 저장
                            self.first_frame_check = True
                            self.ok_image = self.make_show_image(image)

                        # 1. 디텍션 검사
                        if SS.detection_use and self.inspection_count in SS.detection_frame:
                            det_image = cv2.bitwise_and(image, HW.mask_images['det'])
                            self.detection_inspection(det_image)

                        # 2. 분류 검사
                        self.classification_inspection(image)
                    else:
                        time.sleep(0.1)
                
                elif self.state == 'ANALYZING':
                    with self.lock:
                        result = self.analysis_result()
                        self.send_result(result)
                        threading.Thread(target=self.check_and_cleanup).start()
                        self.state = 'WAITING'
                
                elif self.state == 'WAITING':
                    time.sleep(0.1)
                    continue

            except:
                LM.log_print(f"[Inspection] Inspection Thread Error: {traceback.format_exc()}")
                continue

    def detection_inspection(self, image):
        """디텍션 검사"""
        x, y, w, h = SS.detection_coords
        det_image = image[y:y+h, x:x+w]
        self.image_monitor_queue.put(('detection_masked.jpg', det_image))

        if SS.capture_signal_dict["inspection"]:  # 디텍션 검사 이미지 저장
            detection_image_save_path = f"{self.detection_image_save_base_path}/{SS.current_date}/{SS.model_name}"
            self.image_save_queue.put((detection_image_save_path, det_image))

        engrave_count = DT.detection(det_image)

        if SS.ok_engrave != engrave_count:
            print(f"[Detection] Engrave Count NG: {engrave_count}/{SS.ok_engrave}")
            self.detection_ng_count += 1
            
            if self.recent_image_send_count['ENGRAVE']['NG'] < 1:  # 각인 불량 이미지 전송
                SS.send_queue.put(('msg', f'recent:ng:engrave:'))
                SS.send_queue.put(('image', det_image))
                self.recent_image_send_count['ENGRAVE']['NG'] += 1

            if SS.capture_signal_dict["ng"]:  # 불량 이미지 저장
                det_ng_image_save_path = f"{self.ng_image_save_base_path}/detection/{SS.current_date}/{SS.model_name}/{engrave_count}"
                self.image_save_queue.put((det_ng_image_save_path, det_image))
        else:
            print(f"[Detection] Engrave Count OK: {engrave_count}/{SS.ok_engrave}")

            if self.recent_image_send_count['ENGRAVE']['OK'] < 1:  # 각인 양품 이미지 전송
                SS.send_queue.put(('msg', f'recent:ok:engrave:'))
                SS.send_queue.put(('image', det_image))
                self.recent_image_send_count['ENGRAVE']['OK'] += 1

        if len(SS.detection_frame) == self.detection_ng_count:
            self.ng_name = 'detection'
            self.det_ng_image = self.make_show_image(image)
    
    def rotate_image(self, image, angle):
        """이미지 회전"""
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(image, M, (cols, rows))
    
    def get_mask_image(self, mask_type):
        """마스킹 타입에 따른 마스킹 이미지 반환"""
        if mask_type == 'detection':
            return self.mask_images.get('det')
        elif mask_type == '_sub':
            return self.mask_images.get('sub')
        else:
            return self.mask_images.get('main')
    
    def classification_inspection(self, image):
        """분류 검사"""
        ori_image = image.copy()
        if SS.recent_origin_send_signal and self.recent_image_send_count['ORIGIN'] < 1:  # 원본 이미지 1장 전송
            SS.send_queue.put(('msg', f'recent:origin::'))
            SS.send_queue.put(('image', ori_image))
            self.recent_image_send_count['ORIGIN'] += 1

        for idx, (name, values) in enumerate(SS.inspection_coords.items()):
            part_name = name
            mask_type, x, y, w, h, angle = values[0], values[1], values[2], values[3], values[4], values[5]
            
            # 딕셔너리에서 마스킹 이미지 가져오기
            mask_image = self.get_mask_image(mask_type)
            
            if mask_image is not None:
                # 마스킹 적용
                masked_image = cv2.bitwise_and(image, mask_image)
                mask_suffix = 'sub' if mask_type == '_sub' else 'main'
                self.image_monitor_queue.put((f'{part_name}_classi_{mask_suffix}_masked.jpg', masked_image))
            else:
                # 마스킹이 없는 경우 원본 이미지 사용
                masked_image = image
                self.image_monitor_queue.put((f'{part_name}_no_mask.jpg', masked_image))

            cropped_image = masked_image[y:y+h, x:x+w]
            rotated_image = self.rotate_image(cropped_image, angle)
            
            if SS.capture_signal_dict["inspection"]:  # 분류 검사 이미지 저장
                classification_image_save_path = f"{self.classification_image_save_base_path}/{SS.current_date}/{SS.model_name}/{part_name}"
                self.image_save_queue.put((classification_image_save_path, rotated_image))
            
            self.image_monitor_queue.put((f'{part_name}_classi_inspection.jpg', rotated_image))
            label, amount = CL.classification(rotated_image)
            print(label, amount)
            
            if 'OK' not in label.upper():
                if SS.capture_signal_dict["ng"]:
                    ng_image_save_path = f"{self.ng_image_save_base_path}/classification/{SS.current_date}/{SS.model_name}/{part_name}"
                    self.image_save_queue.put((ng_image_save_path, rotated_image))

                if self.recent_image_send_count[part_name]['NG'] < 2:  # NG 2장 전송
                    SS.send_queue.put(('msg', f'recent:ng:{part_name}:{label}'))
                    SS.send_queue.put(('image', rotated_image))
                    self.recent_image_send_count[part_name]['NG'] += 1
            else:
                if self.recent_image_send_count[part_name]['OK'] < 1:  # OK 1장 전송
                    SS.send_queue.put(('msg', f'recent:ok:{part_name}:{label}'))
                    SS.send_queue.put(('image', rotated_image))
                    self.recent_image_send_count[part_name]['OK'] += 1

            self.inspection_result_dict[part_name].append({
                'image': ori_image,
                'coords': (x, y, w, h),
                'label': label,
                'amount': amount
            })
    
    def analysis_result(self):
        """검사 결과 분석"""
        final_results = {}
        
        print(f"\n{'='*54}\n{'검사 결과 종합':^54}\n{'='*54}")
        
        # 각 파트별로 개별 테이블 생성
        try :
            # 1. 디텍션 검사 결과 먼저 확인
            if SS.detection_use and self.ng_name == 'detection':
                final_results['detection'] = {
                    'result': 'NG',
                    'ng_type': 'DETECTION',
                    'ng_type_name': '각인 검사 불량',
                    'ng_image': self.det_ng_image
                }
                print(f"\n[detection]")
                print("-" * 54)
                print("각인 검사 결과: NG")
                print("="*54)
                return final_results
            
            # 2. 분류 검사 결과 확인
            for part_name, results in self.inspection_result_dict.items():
                if results == []:  # 검사 오류로 검사 결과가 없는 경우
                    final_results[part_name] = {
                        'result': 'NG',
                        'ng_type': 'error',
                        'ng_type_name': '검사 오류',
                        'ng_image': None
                    }
                    return final_results

                print(f"\n[{part_name}]")
                print("-" * 54)
                
                # 헤더 출력
                headers = ["#", "label", "amout", "consecutive", "cumulative"]
                
                # 데이터 준비
                table_data = []
                label_counts = {}
                consecutive_counts = {}
                ng_frames = {}
                current_label = None
                current_count = 0
                
                for frame_idx, result in enumerate(results):
                    row_data = [f"{frame_idx+1:2d}"]
                    
                    if result['label'] in SS.label_ng_conditions:
                        amount_limit = SS.label_ng_conditions[result['label']][1]
                        
                        if result['amount'] > amount_limit:
                            label_counts[result['label']] = label_counts.get(result['label'], 0) + 1
                            if result['label'] not in ng_frames:
                                ng_frames[result['label']] = []
                            ng_frames[result['label']].append((frame_idx, result['image'], result['coords']))
                            
                            if result['label'] == current_label:
                                current_count += 1
                            else:
                                current_count = 1
                            
                            consecutive_counts[result['label']] = max(consecutive_counts.get(result['label'], 0), current_count)
                            row_data.extend([f"{result['label']}*", f"{result['amount']:3d}*", f"{current_count}", f"{label_counts[result['label']]}"])
                        else:
                            current_count = 0
                            row_data.extend([result['label'], f"{result['amount']:3d}", "0", str(label_counts.get(result['label'], 0))])
                        
                        current_label = result['label']
                    else:
                        current_label = None
                        current_count = 0
                        row_data.extend([result['label'], f"{result['amount']:3d}", "-", "-"])
                    
                    table_data.append(row_data)
                
                # 테이블 출력
                print(tabulate(table_data, headers=headers, tablefmt="simple"))
                
                # 파트별 최종 판정
                final_ng = None
                ng_image = None
                ng_coords = None
                
                for label, counts in label_counts.items():
                    if label in SS.label_ng_conditions:
                        name, _, consec_limit, total_limit = SS.label_ng_conditions[label]
                        max_consec = consecutive_counts.get(label, 0)
                        if max_consec >= consec_limit or counts >= total_limit:
                            final_ng = label
                            frame_idx = ng_frames[label][0][0]  # 첫 번째 불량 프레임의 인덱스
                            ng_image = results[frame_idx]['image']  # 해당 프레임의 이미지
                            ng_coords = results[frame_idx]['coords']  # 해당 프레임의 좌표
                            ng_image = self.make_ng_image(ng_image, ng_coords)  # 불량 영역 표시
                            ng_image = self.make_show_image(ng_image)
                            break
                
                final_results[part_name] = {
                    'result': 'NG' if final_ng else 'OK',
                    'ng_type': final_ng,
                    'ng_type_name': SS.label_ng_conditions[final_ng][0] if final_ng else None,
                    'ng_image': ng_image if ng_image is not None else None
                }
                
                # 파트별 판정 결과 출력
                print(f"\n판정 결과: ", end="")
                if final_ng:
                    ng_name = final_results[part_name]['ng_type_name']
                    print(f"NG - {final_ng} ({ng_name})")
                else:
                    print("OK")
                print("\n" + "="*54)
            
            return final_results

        except:
            LM.log_print(f"[Analysis] Analysis Result Error: {traceback.format_exc()}")
            return {}
    
    def make_ng_image(self, image, coords):
        """이미지에 불량 좌표 표시"""
        x, y, w, h = coords
        image = cv2.rectangle(image.copy(), (x, y), (x+w, y+h), (41, 41, 252), 2)
        return image

    def make_show_image(self, image):
        """출력 이미지 좌표로 이미지 잘라내기"""
        coord = SS.show_coord
        x, y, w, h = coord[0], coord[1], coord[2], coord[3]
        show_image = image[y:y+h, x:x+w]  
        show_image = cv2.resize(show_image, (391, 290))
        return show_image
    
    def send_result(self, result):
        """검사 결과 전송
        1. critical NG list에 있는 불량이 있는지 먼저 확인
        2. critical NG가 없다면 첫 번째 일반 NG 확인
        3. 모든 NG가 없다면 OK 전송
        """
        final_part = None
        final_result = 'OK'
        final_image = None

        # 1. Critical NG 찾기
        for part_name, part_result in result.items():
            if (part_result['result'] == 'NG' and part_result['ng_type'] in SS.critical_ng_list):
                final_part = part_name
                final_result = part_result['ng_type']
                final_image = part_result['ng_image']
                break
        
        # 2. Critical NG가 없는 경우 첫 번째 일반 NG 찾기
        if final_result == 'OK':
            for part_name, part_result in result.items():
                if part_result['result'] == 'NG':
                    final_part = part_name
                    final_result = part_result['ng_type']
                    final_image = part_result['ng_image']
                    break
        
        # 3. 모든 NG가 없는 경우 OK 전송
        if final_result == 'OK':
            final_part = 'ALL'
            final_result = 'OK'
            final_image = self.ok_image

        # 결과 데이터 전송
        LM.log_print(f"[SEND] Result: {final_part} - {final_result}")
        SS.send_queue.put(('msg', f'result:{final_part}:{final_result}'))
        SS.send_queue.put(('image', final_image))

    def get_disk_usage_percent(self):
        """특정 경로의 디스크 사용량(%) 조회"""
        st = os.statvfs('/')
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        used_percent = 100 - (free / total * 100)
        return used_percent
    
    def check_and_cleanup(self):
        """디스크 사용량 체크 후 폴더 삭제"""
        try:
            usage_percent = self.get_disk_usage_percent()
            print(f"[DISK] disk usage: {usage_percent}%")
            if usage_percent > 80:
                if any(SS.capture_signal_dict.values()):  # 캡처 신호 중 하나라도 True인 경우에만 False로 설정
                    SS.capture_signal_dict = {"origin": False, "ng": False, "inspection": False}  # 이미지 저장 신호 모두 False로 설정
                    LM.log_print("[DISK] Set all values in the Capture Signal Dict to False")
                
                if not self.image_save_queue.empty():  # 이미지 저장 큐가 비어있지않으면 삭제 작업 진행X
                    LM.log_print("[DISK] Cleanup skipped - Image save queue is not empty")
                    return

                for dir_path in ['Capture', 'log']:
                    if os.path.exists(dir_path):
                        try:
                            subprocess.run(['sudo', 'rm', '-rf', dir_path], check=True)
                            LM.log_print(f"[DISK] {dir_path} folder deleted")
                        except:
                            LM.log_print(f"[DISK] Failed to delete {dir_path}: {traceback.format_exc()}")
        except Exception as e:
            LM.log_print(f"[DISK] Disk check and cleanup Error: {traceback.format_exc()}")
            time.sleep(0.5)
    
    def check_gpu_enabled(self):
        """그래픽카드 사용 가능 여부 확인"""
        try:
            gpu_available = tf.test.is_gpu_available()
            SS.send_queue.put(('msg', f'gpu_available:{gpu_available}'))
        except Exception as e:
            LM.log_print(f"[GPU] GPU 확인 중 오류 발생: {e}")
            SS.send_queue.put(('msg', f'gpu_available:False'))
    
    def image_monitor_thread(self):
        """이미지 모니터링 스레드:
        폴더에 이미지를 저장해서 유저가 이미지 실시간 확인""" 
        while True:
            if not self.image_monitor_queue.empty():
                try:
                    file_name, image = self.image_monitor_queue.get()

                    if not os.path.isdir('monitor'):
                        os.makedirs('monitor')
                
                    cv2.imwrite(f'monitor/{file_name}', image)
                    # print(f"[ImageMonitor] {file_name} 저장 완료")
                except:
                    LM.log_print(f"[ImageMonitor] Image Monitor Thread Error: {traceback.format_exc()}")
            else:
                time.sleep(0.1)

    def image_save_thread(self):
        """이미지 저장 스레드"""
        MAX_RETRIES = 3  # 최대 재시도 횟수
        
        while True:
            if not self.image_save_queue.empty():
                try:
                    save_path, image = self.image_save_queue.get()
                    retry_count = 0
                    
                    while retry_count < MAX_RETRIES:
                        try:
                            os.makedirs(save_path, exist_ok=True)
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            file_name = f"{timestamp}.jpg"
                            
                            full_path = os.path.join(save_path, file_name)
                            cv2.imwrite(full_path, image)
                            # print(f"[ImageSave] {file_name} 저장 완료")
                            break
                            
                        except OSError as e:
                            retry_count += 1
                            if retry_count >= MAX_RETRIES:
                                LM.log_print(f"[SAVE] Failed to save image after {MAX_RETRIES} attempts. Error: {str(e)}")
                                break
                            LM.log_print(f"[SAVE] Attempt {retry_count}/{MAX_RETRIES} failed: {str(e)}")
                            time.sleep(0.5)
                            
                except Exception as e:
                    LM.log_print(f"[SAVE] Image Save Thread Error: {str(e)}\n{traceback.format_exc()}")
                    time.sleep(0.5)
            else:
                time.sleep(0.1)

    def code_update(self):
        """Client.py 코드 업데이트"""
        def on_rm_error(func, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        
        LM.log_print('[UPDATE] Client code update order')
        repo_name = "Client"
        git_url = f'https://github.com/KRThor/{repo_name}.git'
        tmp_dir = "update_tmp"
        try:
            # 1. 임시 폴더에 다운로드
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, onerror=on_rm_error)
            git.Repo.clone_from(git_url, tmp_dir)

            # 2. 새 Client.py가 정상적으로 존재하는지 검증
            new_client_path = os.path.join(tmp_dir, "Client.py")
            if not os.path.exists(new_client_path):
                LM.log_print(f"[UPDATE] {repo_name} code not found")
                return

            # 3. 기존 Client.py 백업
            if os.path.exists('Client_.py'):
                os.remove('Client_.py')
            if os.path.exists('Client.py'):
                os.rename('Client.py', 'Client_.py')

            # 4. 새 Client.py로 교체
            shutil.move(new_client_path, 'Client.py')
            shutil.rmtree(tmp_dir, onerror=on_rm_error)

            py_compile.compile("Client.py", cfile="Client.pyc")
            LM.log_print(f"[UPDATE] {repo_name} code compile success")
        except Exception as e:
            LM.log_print(f"[UPDATE] {repo_name} code update failed: {traceback.format_exc()}")


if __name__ == "__main__":
    mode = 'game'  # war or game
    LM = LogManager()
    SS = SocketServer()
    threading.Thread(target=SS.send_thread, daemon=True).start()
    GC = GalaxyCamera(mode, LM)
    threading.Thread(target=GC.run, daemon=True).start()
    CL = Classification(LM)
    DT = Detection(LM)
    HW = HardWork()
    threading.Thread(target=HW.inspection_thread, daemon=True).start()
    Process(target=HW.image_monitor_thread, daemon=True).start()
    Process(target=HW.image_save_thread, daemon=True).start()
    
    while True:
        try:
            SS.connect_to_server()
            SS.run()
            time.sleep(1)
        except:
            LM.log_print(f"[MAIN] Main Thread Error: {traceback.format_exc()}")
            time.sleep(1)
