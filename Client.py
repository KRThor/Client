# 24.02.26
from ast import alias
from datetime import datetime
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import traceback
import logging
import pickle
import socket
import copy
import time
import threading
from queue import Queue
from multiprocessing import Process
from multiprocessing import Queue as MQueue
from keras.preprocessing.image import img_to_array
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
# from pypylon import pylon
import gxipy as gx  #갤럭시 카메라 모듈
import numpy as np
from object_detection.utils import label_map_util
import cv2
import imutils
import math
import tensorflow.compat.v1 as tf
import gc
import configparser
import json
import sys
import git
import shutil

try:
    os.system("sudo ifmetric enp4s0 30000")
    os.system("sudo ifmetric enp42s0 100")
except:
    pass

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

if not os.path.exists("log"):
    os.makedirs("log")

nowtime = datetime.now().strftime("%Y_%m_%d")

file_handler = logging.FileHandler(f"log/log_{nowtime}.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

path = os.path.dirname(os.path.abspath(__file__)).split("/")
print(path)
LINE = path[-1].split('_')[2] # e.g. S11
print(LINE)
print(path[-1].split('_')[1])
if path[-1]:
    if path[-1].split('_')[1] == "0":
        if "CUP" in path[-1]:
            HOST = "192.168.0.100"
            PORT = 9999
            CodeSetup = "CUP"
            ClientSetup = '104'
            
        else:
            HOST = "192.168.0.110"
            PORT = 9999
            CodeSetup = "CONE"
            ClientSetup = '114'
        ClientName = "CLIENT0"
        PickleSetup = "PICKLE0"
        CompleteSignal = "COMP0"
        BadtypeCount = 8
    if path[-1].split('_')[1] == "1":
        if "CUP" in path[-1]:
            HOST = "192.168.0.100"
            PORT = 9999
            CodeSetup = "CUP"
            ClientSetup = '101'
        else:
            HOST = "192.168.0.110"
            PORT = 9999
            CodeSetup = "CONE"
            ClientSetup = '111'
        ClientName = "CLIENT1"
        PickleSetup = "PICKLE1"
        CompleteSignal = "COMP1"
        BadtypeCount = 7
    if path[-1].split('_')[1] == "2":
        if "CUP" in path[-1]:
            HOST = "192.168.0.100"
            PORT = 9999
            CodeSetup = "CUP"
            BadtypeCount = 9
            ClientSetup = '102'
        else:
            HOST = "192.168.0.110"
            PORT = 9999
            CodeSetup = "CONE"
            BadtypeCount = 9
            ClientSetup = '112'
        ClientName = "CLIENT2"
        PickleSetup = "PICKLE2"
        CompleteSignal = "COMP2"
    if path[-1].split('_')[1] == "3":
        if "CUP" in path[-1]:
            HOST = "192.168.0.100"
            PORT = 9999
            CodeSetup = "CUP"
            BadtypeCount = 13
            ClientSetup = '103'
        else:
            HOST = "192.168.0.110"
            PORT = 9999
            CodeSetup = "CONE"
            BadtypeCount = 10
            ClientSetup = '113'
        ClientName = "CLIENT3"
        PickleSetup = "PICKLE3"
        CompleteSignal = "COMP3"
    if path[-1].split('_')[1] == "4":
        if "CUP" in path[-1]:
            HOST = "192.168.0.100"
            PORT = 9999
            CodeSetup = "CUP"
            BadtypeCount = 11
            ClientSetup = '105'
        else:
            HOST = "192.168.0.110"
            PORT = 9999
            CodeSetup = "CONE"
            BadtypeCount = 11
            ClientSetup = '115'
        ClientName = "CLIENT4"
        PickleSetup = "PICKLE4"
        CompleteSignal = "COMP4"
    if path[-1].split('_')[1] == "5":
        if "CUP" in path[-1]:
            HOST = "192.168.0.100"
            PORT = 9999
            CodeSetup = "CUP"
            BadtypeCount = 1
            ClientSetup = '100'
        else:
            HOST = "192.168.0.110"
            PORT = 9999
            CodeSetup = "CONE"
            BadtypeCount = 12
            ClientSetup = '116'
        ClientName = "CLIENT5"
        PickleSetup = "PICKLE5"
        CompleteSignal = "COMP5"

# LINE = "SS11"
##testMode
# testMode = True
# if testMode == True:
#     HOST = "192.168.50.53"
#     PORT = 9999
#     CodeSetup = "CUP"
#     BadtypeCount = 8
#     ClientSetup = '102'
#     ClientName = "CLIENT2"
#     PickleSetup = "PICKLE2"
#     CompleteSignal = "COMP2"

class SocketCommunication:
    def __init__(self):
        self.inspectionSession = False
        self.resultSession = False
        self.nowModel = ""
        self.ModelBackup = ""
        # self.myCam = CamNumber
        self.re_pickleRecvData = ''

    def connectTry(self):
        try:
            self.client_socket.close()
        except:
            pass
        print("Notice : [Socket Connecting. wait please]")
        logger.info(f"[Notice] 소켓 연결 시도 - {HOST}, {PORT}")
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((HOST, PORT))
        print("Notice : [Socket Connected]")
        logger.info(f"[Notice] 소켓 연결 완료 - {HOST}, {PORT}")
        msg = ClientName
        self.myName = PickleSetup
        self.client_socket.send((msg.ljust(20)).encode())
        self.inspectionSession = False
        self.resultSession = False

    def pickleLoad(self):
        pass

    def pickleSave(self):
        try:
            filepath = f"models/{EFFI.ProductData[Socket_main.loadModelIndex-1]}"

            try:
                if not (os.path.isdir(filepath)):
                    os.makedirs(os.path.join(filepath))
            except OSError as e:
                logger.info(f"Warning : 해당 문제로 인하여 에러가 발생 - {e}")
                # pass

            with open("{}/InsValue.pickle".format(filepath), "wb") as file:
                pickle.dump(EFFI.ModelSetupList, file)

            print("피클파일 저장 완료 : ", filepath)
        except Exception as ex:
            logger.info(f"Warning : 피클파일 저장 에러 발생 - {ex}")

    def ClientSocketSend(self, data, length=20):  # RESULT, OK, NG, 2463
        if isinstance(data, str):
            self.client_socket.send(data.ljust(length).encode())
        else:
            self.client_socket.send(data)

    def recvall(self, sock, count):
        buf = b""
        while count:
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def retry_procedure(self):  # 프로그램 재실행 OH
        text = '[ ★ ] 프로그램 재시작'
        logger.info(text)
        print(text)
        python = sys.executable
        os.execl(python, python, * sys.argv)

    def make_safe_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def git_clone(self, git_url):
        target_dir = os.getcwd()  # 현재 경로
        self.make_safe_dir(target_dir)
        git.Git(target_dir).clone(git_url)

    def run(self):
        # 키보드로 입력한 문자열을 서버로 전송하고
        # 서버에서 에코되어 돌아오는 메시지를 받으면 화면에 출력
        # quit를 입력할 때 까지 반복
        while True:
            try:
                try:
                    data = self.recvall(self.client_socket, 20)
                except:
                    logger.info(f'Client Socket Connection Error : {traceback.format_exc()}')
                    time.sleep(1)

                if data == None:
                    break

                try:
                    recvDATA = data.decode().strip()
                    # print('Received from the server :',repr(recvDATA))
                except:
                    recvDATA = ""

                print("recvDATA : ", recvDATA)
                logger.info(f"[Notice] Recv Data from Server : {recvDATA}")

                if recvDATA == "REBOOT": # 컴퓨터 재실행 → 프로그램 재실행으로 변경
                    logger.info("[Notice] Reboot Control Signal Recv")
                    # self.retry_procedure() 
                    os.system("sudo reboot")

                elif recvDATA == "WAITTING":
                    logger.info("[NOTICE] WAITTING Signal Recv (pass)")
                    # self.inspectionSession = False
                    # self.resultSession = False

                elif recvDATA == "UPDATE":
                    logger.info("[NOTICE] CODE UPDATE Signal Recv")
                    print("[★] 코드 업데이트 진행")
                    try :
                        repo_name = f"Client"
                        if os.path.exists('Client_.py'):         
                            os.remove("Client_.py")     # 기존 백업 클라이언트 코드 삭제

                        if os.path.exists('Client.py'): # 기존 클라이언트 백업         
                            os.rename('Client.py', 'Client_.py') 
                            
                        git_url = f'https://github.com/KRThor/{repo_name}.git' # 코드 업데이트할 깃허브 주소
                        self.git_clone(git_url) # 다운로드
                        print("[★] 코드 다운로드")
                        time.sleep(0.2)
                        os.rename(f'{repo_name}/Client.py', 'Client.py') # 다운로드 받은 코드 경로 수정
                        time.sleep(0.2)
                        shutil.rmtree(repo_name) # 다운로드 받은 코드 제거 (파일 있으면 충돌로 에러나서 업데이트 후 삭제)
                        print("[★] 폴더 제거")
                        os.system('python compile_C.py')
                        print("[★] 코드 컴파일")
                        print('[★] 코드 업데이트 성공')
                        logger.info(f"[Notice] 코드 업데이트 성공")
                    except :
                        print("[★] 코드 다운로드 실패")
                        logger.info(f"[Notice] 코드 업데이트 실패")
                        print(traceback.format_exc())


                elif recvDATA == "START":
                    if CTH.BypassMode == False:
                        logger.info("[Notice] Inspection Start Signal Recv")
                        CTH.qimageCount = 0
                        CTH.Qimage = Queue()
                        ODC.badSearchCheck = []
                        EFFI.DetClassiResult = []
                        EFFI.resultImageData = [None] * EFFI.PartCounting
                        EFFI.Det_resultLabel = ['NG', 0]
                        ODC.inspectionCheck = False
                        self.inspectionSession = True
                        EFFI.resultPartData = [0] * EFFI.PartCounting

                        EFFI.BadCheckCount = [0] * EFFI.PartCounting
                        EFFI.MissCount = 0
                        EFFI.continuityDetectState = False
                        EFFI.continuityDetectImage = None
                        EFFI.LastBadImage = None
                        # EFFI.Definition_Result = ''

                        # '불량 유형' : 전송될 유형 숫자
                        EFFI.CriticalProductImageDict = {'MISS' : None, 'ROLLER' : None, 'CRACK' : None, 'MIX' : None}
                        # '불량 유형' : 이미지
                        EFFI.CriticalProductResultDict = {'MISS' : [None, 0, 0, 0], 'ROLLER' : [None, 0, 0, 0], 'CRACK' : [None, 0, 0, 0], 'MIX' : [None, 0, 0, 0]}

                        # 검증 스레드 실행
                        t = threading.Thread(target=ODC.inspectionIMG)
                        t.daemon = True
                        t.start()
                    else:
                        logger.info(f"[Notice] Start Recv in Bypass")

                elif recvDATA == "RESULTREQUEST":
                    if CTH.BypassMode == False:
                        logger.info("[Notice] Inspection Result Requast Signal Recv")
                        self.inspectionSession = False
                        self.resultSession = True
                    else:
                        logger.info(f"[Notice] Resultrequest Recv in Bypass")

                elif "MODEL" in recvDATA:
                    self.nowModel = recvDATA
                    if self.ModelBackup != self.nowModel:
                        ODC.ModelLoadComp = False
                        self.ModelBackup = self.nowModel
                        logger.info(f"[Notice] {self.nowModel} Loading Signal Recv")
                        modelIndex = self.nowModel.replace("MODEL", "")
                        self.loadModelIndex = int(modelIndex)
                        EFFI.load_models(EFFI.ProductData[self.loadModelIndex-1])
                        # OH
                        # 외륜 104, 냬륜 111 각인 디텍션 모델 로드
                        if (CodeSetup == 'CUP' and ClientName == "CLIENT0") or (CodeSetup == 'CONE' and ClientName == "CLIENT1") or (LINE == 'SS11' and CodeSetup == 'CUP' and ClientName == "CLIENT1" and (self.nowModel == 'MODEL8' or self.nowModel == 'MODEL7')): 
                            EFFI.Det_load_models(EFFI.ProductData[self.loadModelIndex-1])
                        #     EFFI.DetClassi_load_models(EFFI.ProductData[self.loadModelIndex-1])

                        Socket_main.ClientSocketSend(CompleteSignal)
                        ODC.ModelLoadComp = True
                    else:
                        print(f"[Notice] {self.nowModel} Loading Signal Recv, SameModel Setup (pass)")
                        logger.info(f"[Notice] {self.nowModel} Loading Signal Recv, SameModel Setup (pass)")
                        Socket_main.ClientSocketSend(CompleteSignal)
                        ODC.ModelLoadComp = True

                elif "SETTING" in recvDATA:
                    logger.info("[Notice] Setting Data Requast Signal Recv")
                    if ODC.ModelLoadComp == True:
                        self.ClientSocketSend(self.myName)
                        sendData = str(EFFI.ModelSetupList)
                        self.ClientSocketSend(sendData, 500)

                # 수동 상태에서 수치조절
                elif "MPICKLE" in recvDATA: 
                    if recvDATA[1:] == self.myName:
                        print('[INFO] Manual mode setting data update')
                        logger.info("[INFO] Manual mode setting data update")
                        DataIndex = int(recvDATA[7:8])
                        self.ClientSocketSend(f"DATAREQUEST{DataIndex}")
                        pickleRecvData = self.recvall(self.client_socket, 500)
                        pickleRecvData = pickleRecvData.decode("utf-8")
                        EFFI.ModelSetupList = eval(pickleRecvData)
                        print("수신받은 data : ", EFFI.ModelSetupList)
                        EFFI.checkValueDictUpdate(EFFI.ModelSetupList)
                        print("결과 data : ", EFFI.checkValueDict)
                        self.pickleSave()
                
                # 자동 상태에서 수치조절
                elif "APICKLE" in recvDATA: 
                    if recvDATA[1:] == self.myName:
                        print('[INFO] Auto mode setting data update')
                        logger.info("[INFO] Auto mode setting data update")
                        DataIndex = int(recvDATA[7:8])
                        self.ClientSocketSend(f"DATAREQUEST{DataIndex}")
                        self.re_pickleRecvData = self.recvall(self.client_socket, 500)
                        self.re_pickleRecvData = self.re_pickleRecvData.decode("utf-8")
                        EFFI.reload_setting = True

                elif 'CAPORI' in recvDATA:
                    if ClientSetup in recvDATA:
                        if 'ON' in recvDATA:
                            logger.info("[Notice] Capture(ORI) On Signal Recv")
                            print("[Notice] Capture(ORI) On Signal Recv")
                            CTH.CaptureMode_Ori = True
                        elif 'OFF' in recvDATA:
                            logger.info("[Notice] Capture(ORI) Off Signal Recv")
                            print("[Notice] Capture(ORI) Off Signal Recv")
                            CTH.CaptureMode_Ori = False
                    else:
                        pass
                
                elif 'CAPBAD' in recvDATA:
                    if ClientSetup in recvDATA:
                        if "ON" in recvDATA:
                            logger.info("[Notice] Capture(NG) On Signal Recv")
                            print("[Notice] Capture(NG) On Signal Recv")
                            ODC.NgCaptureCheck = True
                        elif "OFF" in recvDATA:
                            logger.info("[Notice] Capture(NG) Off Signal Recv")
                            print("[Notice] Capture(NG) Off Signal Recv")
                            ODC.NgCaptureCheck = False
                    else:
                        pass

                elif 'CAPINS' in recvDATA:
                    if ClientSetup in recvDATA:
                        if "ON" in recvDATA:
                            logger.info("[Notice] Capture(PROCESS) On Signal Recv")
                            print("[Notice] Capture(PROCESS) On Signal Recv")
                            CTH.CaptureMode_Process = True
                        elif "OFF" in recvDATA:
                            logger.info("[Notice] Capture(PROCESS) Off Signal Recv")
                            print("[Notice] Capture(PROCESS) Off Signal Recv")
                            CTH.CaptureMode_Process = False
                    else:
                        pass

                if 'BYPASS' in recvDATA:
                    if ClientSetup in recvDATA:
                        if 'ON' in recvDATA:
                            logger.info("[Notice] Bypass On Signal Recv")
                            print("[Notice] Bypass On Signal Recv")
                            CTH.BypassMode = True
                        elif 'OFF' in recvDATA:
                            logger.info("[Notice] Bypass Off Signal Recv")
                            print("[Notice] Bypass Off Signal Recv")
                            CTH.BypassMode = False
                    else:
                        pass
                
                if 'PARAMETER' in recvDATA:
                    EFFI.Parameter_Load()
                    EFFI.SettingFile_Checker(EFFI.ProductData[Socket_main.loadModelIndex-1])

            except Exception as ex:
                logger.info(f"Warning : 소켓통신 프로세스 에러 발생 - {traceback.format_exc()}")
                time.sleep(1)
                raise


class cameraRTSP(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.Qimage = Queue()
        self.CaptureMode_Ori = self.BypassMode = self.CaptureMode_Process = False
        self.imgSaveCount = 0
        self.qimageCount = 0

        basic_cam_path = 'CheckValue/basic_setting.ini'
        alias = 'cam1'
        self.name = alias
        self.basic_cam_path = basic_cam_path
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(basic_cam_path, encoding = 'utf-8')
        self.setting_values = self.config['Setting']
        self.basicsetting = self.config.items('Basic')[0]
        self.valuesetting = self.config.items('Setting')
        self.triggersetting = self.config.items('Trigger')[0][1]
        self.allDone = False

        # 2022-12-09 OH
        self.gpu_is_available = True
        self.remove_disk = False

    def connect_cam(self, basicsetting, valuesetting, triggersetting):
        reconnectCount = 0
        while True:
            try:
                print('\n\n\n\nCamera Connect Try')
                if basicsetting[0] == 'sn':
                    self.device_manager = gx.DeviceManager()
                    self.cam = self.device_manager.open_device_by_sn(basicsetting[1])
                    # self.cam.UserSetDefault.set(gx.GxUserSetEntry.USER_SET0)
                    # self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
                elif basicsetting[0] == 'ip':
                    self.device_manager = gx.DeviceManager()
                    self.cam = self.device_manager.open_device_by_ip(basicsetting[1])
                for i in valuesetting:
                    try:
                        print(i[0], int(i[1]))
                        eval(f'self.cam.{(i[0])}.set({int(i[1])})')
                    except:
                        print("[info] 세팅 불가능한 value", i[0])
                self.cam.UserSetSelector.set(gx.GxUserSetEntry.USER_SET0)
                self.cam.UserSetLoad.send_command()
                if triggersetting.upper() == 'OFF':
                    self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
                else: #TriggerMode ON
                    self.cam.TriggerMode.set(gx.GxSwitchEntry.ON)
                    self.cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
                self.cam.stream_on()
                print('Camera Connect Complete')
                break
            except:
                reconnectCount+=1
                print(f'Reconnect Count - {reconnectCount}')
                print(traceback.format_exc())
                time.sleep(1)

    def stream_mode_on(self):
        self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
        self.cam.stream_on()

    def run(self):
        while True:
            CameraErrorCount = 0
            count = 0
            try:
                self.connect_cam(self.basicsetting, self.valuesetting, self.triggersetting)
                while True:
                    Ftime = time.time()

                    raw_image = self.cam.data_stream[0].get_image()
                    count += 1
                    if raw_image is None:
                        print("Getting image failed.")
                        CameraErrorCount += 1
                        logger.info(f"Getting image failed. {count}Frame / {CameraErrorCount}Count")
                        logger.info(traceback.format_exc())
                        if CameraErrorCount >= 3:
                            self.cam.close_device()
                            logger.info(f"Getting image failed Check Over. {count}Frame / {CameraErrorCount}Count")
                            break
                        else:
                            continue

                    rgb_image = raw_image.convert("RGB")
                    if rgb_image is None:
                        print("Image Convert Error")
                        CameraErrorCount += 1
                        logger.info(f"Image Convert Error. {count}Frame / {CameraErrorCount}Count")
                        logger.info(traceback.format_exc())
                        if CameraErrorCount >= 3:
                            self.cam.close_device()
                            logger.info(f"Image Convert Error Check Over. {count}Frame / {CameraErrorCount}Count")
                            break
                        else:
                            continue

                    numpy_image = rgb_image.get_numpy_array()
                    if numpy_image is None:
                        print("To Cv2 Image Error")
                        CameraErrorCount += 1
                        logger.info(f"To Cv2 Image Error. {count}Frame / {CameraErrorCount}Count")
                        logger.info(traceback.format_exc())
                        if CameraErrorCount >= 3:
                            self.cam.close_device()
                            logger.info(f"To Cv2 Image Error Check Over. {count}Frame / {CameraErrorCount}Count")
                            break
                        else:
                            continue
                    CameraErrorCount = 0
                    # print(count, 'Frame : ', time.time() - Ftime)

                    if Socket_main.inspectionSession == True:
                        # ImageContinueCount += 1
                        # if ImageContinueCount <= 3:
                        #     continue
                        # img = cv2.imread('test.jpg')

                        self.qimageCount += 1

                        if self.qimageCount <= 4: # 최초 이미지 버리기
                            continue

                        if CodeSetup == 'CONE' and ClientName == 'CLIENT4':
                            if self.qimageCount % 3 == 1 :
                                continue

                        self.Qimage.put(numpy_image)

                        # print(f"검사 사진 저장 - {self.qimageCount-4}장") # 최초 이미지 버리기

                        if self.CaptureMode_Ori == True:
                            now = datetime.now()
                            year = str(now.year).zfill(4)
                            month = str(now.month).zfill(2)
                            day = str(now.day).zfill(2)
                            hour = str(now.hour).zfill(2)
                            minute = str(now.minute).zfill(2)
                            second = str(now.second).zfill(2)
                            mic = str(now.microsecond).zfill(6)
                            path = f"Capture/{year}_{month}_{day}/{Socket_main.nowModel}/"
                            inputFileName = f'{year}_{month}_{day}_{hour}_{minute}{second}{mic}.jpg'
                            MQ.put((path, inputFileName, numpy_image))

                        if self.qimageCount > 50:
                            logger.info(f"[Notice] Camera Working Over Session")
                            Socket_main.inspectionSession = False
                            # Socket_main.resultSession = True
                        # Socket_main.inspectionSession = False

                    if Socket_main.resultSession == True:
                        # ImageContinueCount = 0
                        checkCount = 0
                        while True:
                            if ODC.inspectionCheck == True:
                                print("[Notice] Inspection Session Break Check")
                                logger.info("[Notice] Inspection Session Break Check")
                                break
                            print("inspection is not finish")
                            time.sleep(0.1)
                            checkCount += 1
                            if checkCount >= 50:
                                print("[Notice] Force Break ResultSession")
                                logger.info("[Notice] Force Break ResultSession")
                                ODC.inspectionCheck = False
                                break

                        Socket_main.resultSession = False

                        SendingImage = []
                        LabelCheck_Value = True
                        if 'MODEL' in Socket_main.nowModel :
                            # 검증 결과 데이터 Print Up
                            print(f'[Result Report]')
                            print(f'[Part Result]')
                            for i in range(EFFI.PartCounting):
                                Limit = EFFI.checkValueDict[f"PART{i+1}"][2]
                                print(f"PART - {i+1} / Count : {EFFI.resultPartData[i]} / Limit : {Limit}")
                            print(f'[Nomal Product Continuity Result] - {EFFI.continuityDetectState}')
                            for CriticalLabel in EFFI.CriticalProductList:
                                if CriticalLabel in EFFI.CriticalProductSetup[ClientName]:
                                    Limit = EFFI.checkValueDict[CriticalLabel][1]
                                    Limit2 = EFFI.checkValueDict[CriticalLabel][2]
                                    print(F'[Critical Bad Product Result] - ({CriticalLabel})\n [Continuity Result] - {EFFI.CriticalProductResultDict[CriticalLabel][0]}\n [Continuity Count] - {EFFI.CriticalProductResultDict[CriticalLabel][2]} / Limit - {Limit}\n [Critical Total Count] - {EFFI.CriticalProductResultDict[CriticalLabel][3]} / Limit - {Limit2}')
                            if (CodeSetup == "CUP" and ClientName == "CLIENT0" and (Socket_main.nowModel in EFFI.Det_ModelSetup['CUP'])) or (CodeSetup == "CONE" and ClientName == "CLIENT1" and (Socket_main.nowModel in EFFI.Det_ModelSetup['CONE'])) or (LINE == 'SS11' and CodeSetup == "CUP" and ClientName == "CLIENT1" and (Socket_main.nowModel == "MODEL7" or Socket_main.nowModel == "MODEL8")):
                                if EFFI.Det_Running == True:
                                    print(f'[Engrave Detection Result]\nEFFI.Det_resultLabel > {EFFI.Det_resultLabel}')

                            NgCheck = True  # True - 불량 체킹 이력 없음, False - 불량 체킹 이력 있음

                            #중대 불량 연속검사 체크
                            for CriticalLabel in EFFI.CriticalProductList:
                                if CriticalLabel in EFFI.CriticalProductSetup[ClientName]:
                                    if EFFI.CriticalProductResultDict[CriticalLabel][0] == False:
                                        print(f'[X] 중대불량 연속검사 발생 - ({CriticalLabel} (불량)) / 연속검사 횟수 - {EFFI.CriticalProductResultDict[CriticalLabel][2]} / 제한 - {EFFI.checkValueDict[CriticalLabel][1]}')
                                        ODC.badSearchCheck.append(EFFI.CriticalProductType[CriticalLabel])
                                        SendingImage.append(EFFI.CriticalProductImageDict[CriticalLabel])
                                        NgCheck = False
                                        break
                                    else:
                                        print(f'[✓] 중대불량 연속검사 양품 - ({CriticalLabel} (양품)) / 연속검사 횟수 - {EFFI.CriticalProductResultDict[CriticalLabel][2]} / 제한 - {EFFI.checkValueDict[CriticalLabel][1]}')
                                        pass
                            
                            #중대 불량 횟수검사 체크
                            if NgCheck == True:
                                for CriticalLabel in EFFI.CriticalProductList:
                                    if CriticalLabel in EFFI.CriticalProductSetup[ClientName]:
                                        if EFFI.CriticalProductResultDict[CriticalLabel][3] >= EFFI.checkValueDict[CriticalLabel][2]:
                                            print(f'[X] 중대불량 횟수검사 초과 발생 ({CriticalLabel} (불량)) / 총 검출 횟수 - {EFFI.CriticalProductResultDict[CriticalLabel][3]} / 제한 - {EFFI.checkValueDict[CriticalLabel][2]}')
                                            ODC.badSearchCheck.append(EFFI.CriticalProductType[CriticalLabel])
                                            SendingImage.append(EFFI.CriticalProductImageDict[CriticalLabel])
                                            NgCheck = False
                                            break
                                        else:
                                            print(f'[✓] 중대불량 횟수검사 양품 - ({CriticalLabel} (양품)) / 총 검출 횟수 - {EFFI.CriticalProductResultDict[CriticalLabel][3]} / 제한 - {EFFI.checkValueDict[CriticalLabel][2]}')
                                            pass

                            #각인 검사 체크
                            if NgCheck == True:
                                if (CodeSetup == "CUP" and ClientName == "CLIENT0" and (Socket_main.nowModel in EFFI.Det_ModelSetup['CUP'])) or (CodeSetup == "CONE" and ClientName == "CLIENT1" and (Socket_main.nowModel in EFFI.Det_ModelSetup['CONE'])) or (LINE == 'SS11' and CodeSetup == "CUP" and ClientName == "CLIENT1" and (Socket_main.nowModel == "MODEL7" or Socket_main.nowModel == "MODEL8")) :
                                    if EFFI.Det_Running == True:
                                        if EFFI.Det_resultLabel[0] == 'NG': # 각인 분류 만 패스
                                            for i in EFFI.DetClassiResult :
                                                print("*", i)
                                            print(f"[X] 각인 누락 불량 - 검출된 갯수 > {str(EFFI.Det_resultLabel[1])}")
                                            # logger.info(f'[Notice] 디텍션 라벨검증 불량 - {str(EFFI.Det_resultLabel[1])}detect')
                                            # ODC.badSearchCheck.append(BadtypeCount)
                                            if EFFI.Det_resultLabel[1] == 0:
                                                LabelCheck_Value = False
                                            else:
                                                LabelCheck_Value = None
                                            # SendingImage.append(EFFI.resultOKImageData)

                                            # 1112 각인 분류 패스
                                            # elif EFFI.Det_resultLabel[0] == 'ClassiNG': # 각인 분류 만 패스
                                            #     for i in EFFI.DetClassiResult :
                                            #         print("*", i)
                                            #     print("[★] 각인 분류 모양 불량")
                                            #     logger.info('[Notice] 각인 분류 불량 판정')
                                            #     LabelCheck_Value = False
                                            #     SendingImage.append(EFFI.resultOKImageData)
                                            SendingImage.append(EFFI.resultOKImageData)
                                            NgCheck = False

                                        else:
                                            # for i in EFFI.DetClassiResult:
                                            #     print("*", i)
                                            EFFI.DetClassiResult = []
                                            print(f"[✓] 각인 양품")
                                            # logger.info(f'[Notice] 디텍션 라벨검증 양품')

                            #일반 파트별 연속검사 체크
                            if NgCheck == True:
                                if EFFI.continuityDetectState == True:
                                    print("[X] 일반 파트별 연속검사 불량 발생")
                                    ODC.badSearchCheck.append(BadtypeCount)
                                    SendingImage.append(EFFI.continuityDetectImage)
                                    NgCheck = False
                                else:
                                    print("[✓] 일반 파트별 연속검사 양품")

                            #일반 파트별 횟수검사 체크
                            if NgCheck == True:
                                for i in range(EFFI.PartCounting):
                                    if EFFI.resultPartData[i] >= EFFI.checkValueDict[f"PART{i+1}"][2]:
                                        Limit = EFFI.checkValueDict[f"PART{i+1}"][2]
                                        print(f"[X] PART - {i+1} 횟수검사 불량 / 검출 횟수 : {EFFI.resultPartData[i]} / 제한 : {Limit}")
                                        ODC.badSearchCheck.append(BadtypeCount)
                                        SendingImage.append(EFFI.resultImageData[i])
                                        NgCheck = False
                                        break
                                    else:
                                        Limit = EFFI.checkValueDict[f"PART{i+1}"][2]
                                        print(f"[✓] PART - {i+1} 횟수검사 (양품) / 검출 횟수 : {EFFI.resultPartData[i]} / 제한 : {Limit}")
                                        continue

                            #일반 파트 + 중대 불량 총 검출 횟수검사 체크
                            if NgCheck == True:
                                TotalNgCounting = 0
                                for i in range(EFFI.PartCounting):
                                    TotalNgCounting += EFFI.resultPartData[i]
                                for CriticalLabel in (EFFI.CriticalProductList):
                                    if CriticalLabel in EFFI.CriticalProductSetup[ClientName]:
                                        TotalNgCounting += EFFI.CriticalProductResultDict[CriticalLabel][3]
                                if TotalNgCounting >= EFFI.checkValueDict["TOTAL"][0]:
                                    print(f"[X] 전체 불량 검출 횟수 초과 (불량) / 전체 불량 검출 횟수 > {TotalNgCounting}개 / 총 검사 제한 > {EFFI.checkValueDict['TOTAL'][0]}")
                                    ODC.badSearchCheck.append(BadtypeCount)
                                    SendingImage.append(EFFI.LastBadImage)
                                    NgCheck = False
                                else:
                                    print(f"[✓] 전체 불량 검출 횟수 미달 (양품) / 전체 불량 검출 횟수 > {TotalNgCounting} / 총 검사 제한 > {EFFI.checkValueDict['TOTAL'][0]}")

                            if NgCheck == True:
                                SendingImage.append(EFFI.resultOKImageData)
                        else:
                            SendingImage.append(EFFI.resultOKImageData)
                        # OH
                        YouCanSeeImage = SendingImage[0].copy()
                        (reY, reX, reS, reS2, reP) = EFFI.axisC[Socket_main.nowModel]
                        YouCanSeeImage = YouCanSeeImage[reY : reY + reS, reX : reX + reS2].copy()

                        if LINE == 'SS9' or LINE == 'SS13':
                            if CodeSetup == 'CUP':
                                if ClientName == "CLIENT2" or ClientName == "CLIENT3":
                                    YouCanSeeImage = cv2.rotate(YouCanSeeImage, cv2.ROTATE_90_CLOCKWISE)

                                elif ClientName == "CLIENT4":
                                    YouCanSeeImage = cv2.rotate(YouCanSeeImage, cv2.ROTATE_90_COUNTERCLOCKWISE)

                            else:
                                if ClientName == "CLIENT2" or ClientName == "CLIENT5":
                                    YouCanSeeImage = cv2.rotate(YouCanSeeImage, cv2.ROTATE_90_CLOCKWISE)

                                elif ClientName == "CLIENT3" or ClientName == "CLIENT4":
                                    YouCanSeeImage = cv2.rotate(YouCanSeeImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        else: 
                            if CodeSetup == 'CUP':
                                if ClientName == "CLIENT2" or ClientName == "CLIENT3":
                                    YouCanSeeImage = cv2.rotate(YouCanSeeImage, cv2.ROTATE_90_COUNTERCLOCKWISE)

                                elif ClientName == "CLIENT4":
                                    YouCanSeeImage = cv2.rotate(YouCanSeeImage, cv2.ROTATE_90_CLOCKWISE)

                            else:
                                if ClientName == "CLIENT2" or ClientName == "CLIENT5":
                                    YouCanSeeImage = cv2.rotate(YouCanSeeImage, cv2.ROTATE_90_COUNTERCLOCKWISE)

                                elif ClientName == "CLIENT3" or ClientName == "CLIENT4":
                                    YouCanSeeImage = cv2.rotate(YouCanSeeImage, cv2.ROTATE_90_CLOCKWISE)

                        (O_H, O_W) = YouCanSeeImage.shape[:2]

                        if len(ODC.badSearchCheck) > 0:  # 불량 이면
                            cv2.rectangle(YouCanSeeImage, (0, 0), (O_W, O_H), (0, 0, 255), 10)
                            SendingImage[0] = YouCanSeeImage.copy()
                            # cv2.imwrite("SendingImage.jpg", SendingImage[0])

                        elif len(ODC.badSearchCheck) == 0:  # 양품 이면
                            cv2.rectangle(YouCanSeeImage, (0, 0), (O_W, O_H), (0, 255, 0), 10)
                            SendingImage[0] = YouCanSeeImage.copy()
                            # cv2.imwrite("SendingImage.jpg", SendingImage[0])
                        (O_H_D, O_W_D) = SendingImage[0].shape[:2]
                        if LabelCheck_Value == True:
                            pass
                        elif LabelCheck_Value == None:
                            ODC.badSearchCheck.append(BadtypeCount)
                            cv2.putText(SendingImage[0], 'Detection Result False', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                            cv2.rectangle(SendingImage[0], (0, 0), (O_W_D, O_H_D), (0, 0, 255), 10)
                        else:
                            logger.info('[Notice] Engrave Result NG - Count 0')
                            ODC.badSearchCheck.append(EFFI.CriticalProductType["ENGRAVE"])
                            cv2.putText(SendingImage[0], 'Detection Result False', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                            cv2.rectangle(SendingImage[0], (0, 0), (O_W_D, O_H_D), (0, 0, 255), 10)
                        # if ClientName in ['CLIENT2', 'CLIENT3', 'CLIENT4', 'CLIENT5']:
                        #     cv2.putText(SendingImage[0], f'{EFFI.Definition_Result}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                        # else:
                        #     cv2.putText(SendingImage[0], f'{EFFI.Definition_Result}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)

                        # 결과 데이터 송신
                        ###################
                        # ODC.badSearchCheck = [0,1,2,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                        ###################

                        # 결과 데이터 송신
                        msg = "0"
                        # img = cv2.imread('test1.jpg')

                        for i in range(1, 14):
                            if i in ODC.badSearchCheck:
                                msg = msg + "1"
                            else:
                                msg = msg + "0"

                        print('Sending Image Count - ', len(SendingImage))

                        # GPU 사용 확인 2022-12-09 OH
                        gpu = tf.test.is_gpu_available()
                        if gpu == True :
                            self.gpu_is_available = True
                            osg = f"{ClientName}GPUOK" 
                            Socket_main.ClientSocketSend(osg)
                        else : 
                            self.gpu_is_available = False
                            print("[★] GPU 미사용")
                            osg = f"{ClientName}GPUNG"
                            Socket_main.ClientSocketSend(osg)
                            
                        msg2 = f"{ClientName}RESULT"
                        Socket_main.ClientSocketSend(msg2)
                        print('Sending Result Data - ', msg)
                        Socket_main.ClientSocketSend(msg, 100)
                        print(f"[Notice] Final Inspection Result Send to Server : {msg}")
                        logger.info(f"[Notice] Final Inspection Result Send to Server : {msg}")

                        EFFI.ok_send_count_list = [0] * EFFI.PartCounting
                        EFFI.ng_send_count_list = [0] * EFFI.PartCounting
                        EFFI.ED_ok_send_count = 0
                        EFFI.ED_ng_send_count = 0
                        
                        # Definition_SendData = [ClientName, EFFI.Definition_Result]
                        # pickleDefPickle = pickle.dumps(Definition_SendData)
                        # Socket_main.ClientSocketSend(str(len(pickleDefPickle)).ljust(20).encode())
                        # Socket_main.ClientSocketSend(pickleDefPickle)
                        # print(f'[Notice] Final Definition Result Send to Server : {Definition_SendData}')
                        # logger.info(f'[Notice] Final Definition Result Send to Server : {Definition_SendData}')

                        pickleData = pickle.dumps(SendingImage)
                        Socket_main.client_socket.send(str(len(pickleData)).ljust(20).encode())
                        Socket_main.client_socket.send(pickleData)
                        Socket_main.resultSession = False
                        print(f"[Notice] Final Inspection ImageData Send to Server")
                        logger.info(f"[Notice] Final Inspection ImageData Send to Server")

                        # 용량 확인 2022-12-09 OH
                        result = self.get_disk_space()
                        if result > 80 :
                            self.remove_disk = True

                        self.allDone = True  # 모든 세션 완료 되었는지 확인하는 변수

                        if (self.allDone == True) and (EFFI.ReloadSignal == True):  # 모델 재로딩 완료, 검사 끝나면 모델 스위칭
                            EFFI.model = EFFI.ReModel
                            EFFI.lb = EFFI.Relb

                            try:
                                del EFFI.ReModel
                            except:
                                pass
                            try:
                                EFFI.Relb
                            except:
                                pass
                            try:
                                gc.collect()
                            except:
                                pass
                            try:
                                K.clear_session()
                            except:
                                pass
                            
                            EFFI.ReloadSignal = False
                            print("[★] 새로운 모델로 변경 완료!")

                        if (self.allDone == True) and (EFFI.reload_setting == True):  # 모델 재로딩 완료, 검사 끝나면 모델 스위칭
                            EFFI.ModelSetupList = eval(Socket_main.re_pickleRecvData)
                            print("수신받은 data : ", EFFI.ModelSetupList)
                            EFFI.checkValueDictUpdate(EFFI.ModelSetupList)
                            print("결과 data : ", EFFI.checkValueDict)
                            Socket_main.pickleSave()

                            EFFI.reload_setting = False
                            print("[★] 새로운 세팅으로 변경 완료!")

            except:
                print(traceback.format_exc())
                self.cam.close_device()
                logger.info(f'Galaxy Camera Process Error - {traceback.format_exc()}')

    def get_disk_space(self): # 2022-12-09 OH
        st = os.statvfs("/")

        # 총, 남은 디스크 용량 계산
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize

        now_total = (total/1024/1024/1024)
        now_free = (free/1024/1024/1024)
        result = 100 - ((now_free / now_total) * 100)
        
        return result

    def remove_forder(self): # OH
        while True :
            try : 
                if self.remove_disk == True:
                    if os.path.exists('Capture'):
                        shutil.rmtree('Capture')
                        self.CaptureMode = False
                        print("[★] Capture 폴더 삭제")

                        result = self.get_disk_space()
                        if result > 80 :
                            if os.path.exists('NgCapture'):
                                shutil.rmtree('NgCapture')
                                print("[★] NgCapture 폴더 삭제")
                        else : 
                            pass
                        
                        self.remove_disk = False
                else :
                    time.sleep(0.5)
            except :
                print(f'{traceback.format_exc()}')
                time.sleep(0.5)


class EfficientNetIMG:
    def __init__(self):
        self.modelResizeDict = {
            "MODEL1": [300, 300],
            "MODEL2": [300, 300],
            "MODEL3": [300, 300],
            "MODEL4": [300, 300],
            "MODEL5": [300, 300],
            "MODEL6": [300, 300],
            "MODEL7": [300, 300],
            "MODEL8": [300, 300],
            "MODEL9": [300, 300],
            "MODEL10": [300, 300],
            "MODEL11": [300, 300],
            "MODEL12": [300, 300],
            "MODEL13": [300, 300],
            "MODEL14": [300, 300],
            "MODEL15": [300, 300],
            "MODEL16": [300, 300],
            "MODEL17": [300, 300],
            "MODEL18": [300, 300],
            "MODEL19": [300, 300],
            "MODEL20": [300, 300],
            "MODEL21": [300, 300],
            "MODEL22": [300, 300],
            "MODEL23": [300, 300],
            "MODEL24": [300, 300],
            "MODEL25": [300, 300],
            "MODEL26": [300, 300],
            "MODEL27": [300, 300],
            "MODEL28": [300, 300],
            "MODEL29": [300, 300],
            "MODEL30": [300, 300]
        }

        if LINE == 'SS8':
            if CodeSetup == 'CUP' :
                self.EngraveCheckLimit = {
                    'MODEL1' : 4,
                    'MODEL2' : 4,
                    'MODEL3' : 5,
                    'MODEL4' : 4,
                    'MODEL5' : 8,
                    'MODEL6' : 4,
                    'MODEL7' : 4,
                    'MODEL8' : 4,
                    'MODEL9' : 4,
                    'MODEL10' : 4,
                    'MODEL11' : 4,
                    'MODEL12' : 4,
                    'MODEL13' : 4,
                    'MODEL14' : 4,
                    'MODEL15' : 4,
                    'MODEL16' : 4,
                    'MODEL17' : 4,
                    'MODEL18' : 4,
                    'MODEL19' : 4,
                    'MODEL20' : 4,
                    'MODEL21' : 4,
                    'MODEL22' : 4,
                    'MODEL23' : 4,
                    'MODEL24' : 4,
                    'MODEL25' : 4,
                    'MODEL26' : 4,
                    'MODEL27' : 4,
                    'MODEL28' : 4,
                    'MODEL29' : 4,
                    'MODEL30' : 4
                }
            elif CodeSetup == 'CONE' :
                self.EngraveCheckLimit = {
                    'MODEL1' : 4,
                    'MODEL2' : 5,
                    'MODEL3' : 4,
                    'MODEL4' : 4,
                    'MODEL5' : 4,
                    'MODEL6' : 4,
                    'MODEL7' : 4,
                    'MODEL8' : 4,
                    'MODEL9' : 4,
                    'MODEL10' : 4,
                    'MODEL11' : 4,
                    'MODEL12' : 4,
                    'MODEL13' : 4,
                    'MODEL14' : 4,
                    'MODEL15' : 4,
                    'MODEL16' : 4,
                    'MODEL17' : 4,
                    'MODEL18' : 4,
                    'MODEL19' : 4,
                    'MODEL20' : 4,
                    'MODEL21' : 4,
                    'MODEL22' : 4,
                    'MODEL23' : 4,
                    'MODEL24' : 4,
                    'MODEL25' : 4,
                    'MODEL26' : 4,
                    'MODEL27' : 4,
                    'MODEL28' : 4,
                    'MODEL29' : 4,
                    'MODEL30' : 4
                }

        elif LINE == 'SS9':
            if CodeSetup == 'CUP' :
                self.EngraveCheckLimit = {
                    'MODEL1' : 4,
                    'MODEL2' : 4,
                    'MODEL3' : 6,
                    'MODEL4' : 4,
                    'MODEL5' : 4,
                    'MODEL6' : 4,
                    'MODEL7' : 4,
                    'MODEL8' : 4,
                    'MODEL9' : 4,
                    'MODEL10' : 5,
                    'MODEL11' : 6,
                    'MODEL12' : 4,
                    'MODEL13' : 4,
                    'MODEL14' : 4,
                    'MODEL15' : 4,
                    'MODEL16' : 4,
                    'MODEL17' : 4,
                    'MODEL18' : 4,
                    'MODEL19' : 4,
                    'MODEL20' : 4,
                    'MODEL21' : 4,
                    'MODEL22' : 4,
                    'MODEL23' : 4,
                    'MODEL24' : 4,
                    'MODEL25' : 4,
                    'MODEL26' : 4,
                    'MODEL27' : 4,
                    'MODEL28' : 4,
                    'MODEL29' : 4,
                    'MODEL30' : 4
                }
            elif CodeSetup == 'CONE' :
                self.EngraveCheckLimit = {
                    'MODEL1' : 4,
                    'MODEL2' : 5,
                    'MODEL3' : 6,
                    'MODEL4' : 6,
                    'MODEL5' : 4,
                    'MODEL6' : 4,
                    'MODEL7' : 4,
                    'MODEL8' : 4,
                    'MODEL9' : 4,
                    'MODEL10' : 4,
                    'MODEL11' : 4,
                    'MODEL12' : 4,
                    'MODEL13' : 4,
                    'MODEL14' : 5,
                    'MODEL15' : 4,
                    'MODEL16' : 4,
                    'MODEL17' : 4,
                    'MODEL18' : 4,
                    'MODEL19' : 4,
                    'MODEL20' : 4,
                    'MODEL21' : 4,
                    'MODEL22' : 4,
                    'MODEL23' : 4,
                    'MODEL24' : 4,
                    'MODEL25' : 4,
                    'MODEL26' : 4,
                    'MODEL27' : 4,
                    'MODEL28' : 4,
                    'MODEL29' : 4,
                    'MODEL30' : 4
                }

        elif LINE == 'SS11':
            if CodeSetup == 'CUP' :
                self.EngraveCheckLimit = {
                    'MODEL1' : 4,
                    'MODEL2' : 4,
                    'MODEL3' : 4,
                    'MODEL4' : 4,
                    'MODEL5' : 4,
                    'MODEL6' : 5,
                    'MODEL7' : 4,
                    'MODEL8' : 4,
                    'MODEL9' : 4,
                    'MODEL10' : 4,
                    'MODEL11' : 4,
                    'MODEL12' : 4,
                    'MODEL13' : 4,
                    'MODEL14' : 6,
                    'MODEL15' : 6,
                    'MODEL16' : 4,
                    'MODEL17' : 4,
                    'MODEL18' : 4,
                    'MODEL19' : 4,
                    'MODEL20' : 4,
                    'MODEL21' : 4,
                    'MODEL22' : 4,
                    'MODEL23' : 4,
                    'MODEL24' : 4,
                    'MODEL25' : 4,
                    'MODEL26' : 4,
                    'MODEL27' : 4,
                    'MODEL28' : 4,
                    'MODEL29' : 4,
                    'MODEL30' : 4
                }
                self.EngraveCheckLimit_101 = {
                    'MODEL1' : 2,
                    'MODEL2' : 2,
                    'MODEL3' : 2,
                    'MODEL4' : 2,
                    'MODEL5' : 2,
                    'MODEL6' : 2,
                    'MODEL7' : 2,
                    'MODEL8' : 2,
                    'MODEL9' : 2,
                    'MODEL10' : 2,
                    'MODEL11' : 2,
                    'MODEL12' : 2,
                    'MODEL13' : 2,
                    'MODEL14' : 2,
                    'MODEL15' : 2,
                    'MODEL16' : 4,
                    'MODEL17' : 4,
                    'MODEL18' : 4,
                    'MODEL19' : 4,
                    'MODEL20' : 4,
                    'MODEL21' : 4,
                    'MODEL22' : 4,
                    'MODEL23' : 4,
                    'MODEL24' : 4,
                    'MODEL25' : 4,
                    'MODEL26' : 4,
                    'MODEL27' : 4,
                    'MODEL28' : 4,
                    'MODEL29' : 4,
                    'MODEL30' : 4
                }
            elif CodeSetup == 'CONE' :
                self.EngraveCheckLimit = {
                    'MODEL1' : 4,
                    'MODEL2' : 4,
                    'MODEL3' : 4,
                    'MODEL4' : 4,
                    'MODEL5' : 4,
                    'MODEL6' : 4,
                    'MODEL7' : 4,
                    'MODEL8' : 4,
                    'MODEL9' : 4,
                    'MODEL10' : 5,
                    'MODEL11' : 4,
                    'MODEL12' : 4,
                    'MODEL13' : 6,
                    'MODEL14' : 6,
                    'MODEL15' : 6,
                    'MODEL16' : 4,
                    'MODEL17' : 4,
                    'MODEL18' : 4,
                    'MODEL19' : 4,
                    'MODEL20' : 4,
                    'MODEL21' : 4,
                    'MODEL22' : 4,
                    'MODEL23' : 4,
                    'MODEL24' : 4,
                    'MODEL25' : 4,
                    'MODEL26' : 4,
                    'MODEL27' : 4,
                    'MODEL28' : 4,
                    'MODEL29' : 4,
                    'MODEL30' : 4
                }

        elif LINE == 'SS12':
            if CodeSetup == 'CONE' :
                self.EngraveCheckLimit = {
                    'MODEL1' : 4,
                    'MODEL2' : 5,
                    'MODEL3' : 4,
                    'MODEL4' : 4,
                    'MODEL5' : 4,
                    'MODEL6' : 4,
                    'MODEL7' : 4,
                    'MODEL8' : 4,
                    'MODEL9' : 4,
                    'MODEL10' : 4,
                    'MODEL11' : 4,
                    'MODEL12' : 4,
                    'MODEL13' : 4,
                    'MODEL14' : 4,
                    'MODEL15' : 4,
                    'MODEL16' : 4,
                    'MODEL17' : 4,
                    'MODEL18' : 4,
                    'MODEL19' : 4,
                    'MODEL20' : 4,
                    'MODEL21' : 4,
                    'MODEL22' : 4,
                    'MODEL23' : 4,
                    'MODEL24' : 4,
                    'MODEL25' : 4,
                    'MODEL26' : 4,
                    'MODEL27' : 4,
                    'MODEL28' : 4,
                    'MODEL29' : 4,
                    'MODEL30' : 4
                }

        elif LINE == 'SS13':
            if CodeSetup == 'CUP' :
                self.EngraveCheckLimit = {
                    'MODEL1' : 4,
                    'MODEL2' : 4,
                    'MODEL3' : 6,
                    'MODEL4' : 5,
                    'MODEL5' : 4,
                    'MODEL6' : 6,
                    'MODEL7' : 4,
                    'MODEL8' : 4,
                    'MODEL9' : 6,
                    'MODEL10' : 4,
                    'MODEL11' : 4,
                    'MODEL12' : 4,
                    'MODEL13' : 4,
                    'MODEL14' : 4,
                    'MODEL15' : 4,
                    'MODEL16' : 4,
                    'MODEL17' : 4,
                    'MODEL18' : 4,
                    'MODEL19' : 4,
                    'MODEL20' : 4,
                    'MODEL21' : 4,
                    'MODEL22' : 4,
                    'MODEL23' : 4,
                    'MODEL24' : 4,
                    'MODEL25' : 4,
                    'MODEL26' : 4,
                    'MODEL27' : 4,
                    'MODEL28' : 4,
                    'MODEL29' : 4,
                    'MODEL30' : 4
                }
            elif CodeSetup == 'CONE' :
                self.EngraveCheckLimit = {
                    'MODEL1' : 4,
                    'MODEL2' : 4,
                    'MODEL3' : 6,
                    'MODEL4' : 5,
                    'MODEL5' : 4,
                    'MODEL6' : 5,
                    'MODEL7' : 4,
                    'MODEL8' : 4,
                    'MODEL9' : 6,
                    'MODEL10' : 4,
                    'MODEL11' : 4,
                    'MODEL12' : 4,
                    'MODEL13' : 4,
                    'MODEL14' : 4,
                    'MODEL15' : 4,
                    'MODEL16' : 4,
                    'MODEL17' : 4,
                    'MODEL18' : 4,
                    'MODEL19' : 4,
                    'MODEL20' : 4,
                    'MODEL21' : 4,
                    'MODEL22' : 4,
                    'MODEL23' : 4,
                    'MODEL24' : 4,
                    'MODEL25' : 4,
                    'MODEL26' : 4,
                    'MODEL27' : 4,
                    'MODEL28' : 4,
                    'MODEL29' : 4,
                    'MODEL30' : 4
                }

        self.BadCheckCount = []  # part 1,2,3,4 (cropBox index 기준)

        self.SessionCheckList = ["PART1", "PART2", "PART3", "PART4", "PART5", "PART6"]
        self.resultImageData = []
        self.resultOKImageData = None
        self.resultPartData = []
        self.DetClassiResult = []
        self.continuityDetectState = False  # 검사 결과 상태 확인
        self.continuityDetectImage = None  # 해당 파트로 불량발생시 업데이트할 이미지
        self.MissCount = 0
        self.Det_resultLabel = ['NG', 0]
        self.DetClassiImage = [] # 각인 분류 이미지 저장 공간
        self.NgDetClassiCount = 0
        self.PartCounting = 1
        self.LastBadImage = None
        # self.Definition_Result = ''
        self.axisC = {}
        self.cropBox = {}
        self.Det_ModelSetup = {}
        self.Det_Running = True
        self.modelN = ''
        self.ReloadSignal = False   # 모든 정보 재로드

        self.reload_setting = False # 세팅 정보 재로드

        self.ok_send_count_list = [0] * self.PartCounting
        self.ng_send_count_list = [0] * self.PartCounting
        self.ED_ok_send_count = 0
        self.ED_ng_send_count = 0

        # 중대불량 프로세스 추가
        self.CriticalProductList = ['MISS', 'ROLLER', 'CRACK', 'MIX']
        # classification 불량 유형
        self.CriticalProductType = {'MISS' : 1, 'ROLLER' : 2, 'ENGRAVE' : 3, 'IDENTIFICATION' : 4, 'CRACK' : 5, 'MIX' : 6}
        # '불량 유형' : 전송될 유형 숫자
        self.CriticalProductImageDict = {'MISS' : None, 'ROLLER' : None, 'CRACK' : None, 'MIX' : None}
        # '불량 유형' : 이미지
        self.CriticalProductResultDict = {'MISS' : [None, 0, 0, 0], 'ROLLER' : [None, 0, 0, 0], 'CRACK' : [None, 0, 0, 0], 'MIX' : [None, 0, 0, 0]}
        # '불량 유형' : [연속검출 유무 (False - 불량, None - 양품), 연속검사 카운팅(초기화), 연속검사 고점 카운팅(비초기화), 총 검출 횟수(비초기화)]
        if CodeSetup == 'CONE':
            self.CriticalProductSetup = {
                'CLIENT0' : ['MIX'],
                'CLIENT1' : ['MIX'],
                'CLIENT2' : [],
                'CLIENT3' : ['ROLLER', 'CRACK'],
                'CLIENT4' : ['CRACK'],
                'CLIENT5' : ['MIX']
            }
        else:
            self.CriticalProductSetup = {
                'CLIENT0' : ['MIX'],
                'CLIENT1' : ['MIX'],
                'CLIENT2' : ['MISS'],
                'CLIENT3' : [],
                'CLIENT4' : [],
            }
        
        self.Parameter_Load()
        self.ProductData = []
        self.product_infoJson_Load()

    def product_infoJson_Load(self):
        if os.path.isfile('CheckValue/product_info.json'):
            with open('CheckValue/product_info.json', 'r') as read_file:
                ReadJsonData = json.load(read_file)
            self.ProductData = ReadJsonData[LINE][CodeSetup][1]
            print(self.ProductData)

    def Parameter_Load(self):
        if os.path.isfile('CheckValue/parameter.json'):
            with open('CheckValue/parameter.json', 'r') as read_file:
                ReadJsonData = json.load(read_file)
            self.axisC = ReadJsonData["axisC"]
            self.cropBox = ReadJsonData["cropBox"]
            self.Det_ModelSetup = ReadJsonData["Det_ModelSetup"]
            self.ClassiMasker = cv2.imread(f"ClassiMask/{CodeSetup}_{ClientName}_{self.modelN}.png")
            self.SubClassiMask = cv2.imread(f"ClassiMask/{CodeSetup}_{ClientName}_{self.modelN}_sub.png")
            try:
                self.PartCounting = len(self.cropBox[Socket_main.nowModel])
                self.ok_send_count_list = [0] * self.PartCounting
                self.ng_send_count_list = [0] * self.PartCounting
                self.reload_models(self.modelN)
            except:
                pass
            logger.info('[Notice] Inspection Parameter Load Complete')
            print(f'[Notice] Inspection Parameter Load Complete - {self.modelN}')

            # print(f'axisC Load - {self.axisC}')
            # print(f'cropBox Load - {self.cropBox}')
            # print(f'Det_ModelSetup - {self.Det_ModelSetup}')
        else:
            logger.info('[Notice] Inspection Parameter Load False - Not File Found')
            pass

    def SettingFile_Checker(self, model_Number):
        settingPath = f"models/{model_Number}/InsValue.pickle"
        self.checkValueDict = {}
        # PartCount = len(self.cropBox[Socket_main.nowModel])
        if os.path.isfile(settingPath) == True:
            modelCheckResult = pickle.loads(open(settingPath, "rb").read())
            print("PART 갯수 : ", len(self.cropBox[Socket_main.nowModel]), "리스트 갯수 : ", len(modelCheckResult))
            if (len(self.cropBox[Socket_main.nowModel])+len(self.CriticalProductSetup[ClientName])) != (len(modelCheckResult) - 1):
                self.ModelSetupList = []
                for i in range(len(self.cropBox[Socket_main.nowModel])):
                    data = [f"PART{i+1}", [90, 2, 3]]
                    self.ModelSetupList.append(data)
                for i in range(len(self.CriticalProductSetup[ClientName])):
                    self.ModelSetupList.append([f"{self.CriticalProductSetup[ClientName][i]}", [90, 2, 3]])
                data = ["TOTAL", [2]]
                self.ModelSetupList.append(data)

                with open(settingPath, "wb") as f:
                    pickle.dump(self.ModelSetupList, f)
                print("파일 존재 / 새로 저장")
            else:
                self.ModelSetupList = copy.deepcopy(modelCheckResult)
                print("기존 셋업 불러오기")
        else:
            self.ModelSetupList = []
            for i in range(len(self.cropBox[Socket_main.nowModel])):
                data = [f"PART{i+1}", [90, 2, 3]]
                self.ModelSetupList.append(data)
            for i in range(len(self.CriticalProductSetup[ClientName])):
                self.ModelSetupList.append([f"{self.CriticalProductSetup[ClientName][i]}", [90, 2, 3]])
            data = ["TOTAL", [2]]
            self.ModelSetupList.append(data)

            with open(settingPath, "wb") as f:
                pickle.dump(self.ModelSetupList, f)
            print("파일 미존재 / 새로 저장")

        self.checkValueDictUpdate(self.ModelSetupList)
        print(self.checkValueDict)

    def load_models(self, model_number):
        self.modelN = model_number
        try:
            del self.model
        except:
            pass
        try:
            self.lb
        except:
            pass
        try:
            gc.collect()
        except:
            pass
        try:
            K.clear_session()
        except:
            pass

        logger.info("[Notice] Model Loading Start")
        modelPath = f"models/{model_number}/model.hdf5"
        picklePath = f"models/{model_number}/model.pickle"
        
        self.model = load_model(modelPath)    #########################################
        self.lb = pickle.loads(open(picklePath, "rb").read())
        # self.graph_classi = tf.get_default_graph()

        self.SettingFile_Checker(model_number)

        self.PartCounting = len(self.cropBox[Socket_main.nowModel])
        self.ok_send_count_list = [0] * self.PartCounting
        self.ng_send_count_list = [0] * self.PartCounting
        try:
            print(f"Masking File Load - ClassiMask/{CodeSetup}_{ClientName}_{model_number}.png")
            self.ClassiMasker = cv2.imread(f"ClassiMask/{CodeSetup}_{ClientName}_{model_number}.png")
            self.SubClassiMask = cv2.imread(f"ClassiMask/{CodeSetup}_{ClientName}_{model_number}_sub.png")
        except:
            print("Classi Masking File Is Nothing")

        logger.info("[Notice] Model Loading Complete")
        # test
        logger.info("[Notice] Model Test Start")
        testimg = cv2.imread("test.jpg")
        self.inspection(testimg, True)
        logger.info("[Notice] Model Test Complete")

    def reload_models(self, model_number): # OH
        try:
            del self.ReModel
        except:
            pass
        try:
            self.lb
        except:
            pass
        try:
            gc.collect()
        except:
            pass
        try:
            K.clear_session()
        except:
            pass

        logger.info("[Notice] Model Loading Start")
        modelPath = f"models/{model_number}/model.hdf5"
        picklePath = f"models/{model_number}/model.pickle"
        
        self.ReModel = load_model(modelPath)    #########################################
        self.Relb = pickle.loads(open(picklePath, "rb").read())

        self.SettingFile_Checker(model_number)
        self.PartCounting = len(self.cropBox[Socket_main.nowModel])
        self.ok_send_count_list = [0] * self.PartCounting
        self.ng_send_count_list = [0] * self.PartCounting
        try:
            print(f"Masking File Load - ClassiMask/{CodeSetup}_{ClientName}_{model_number}.png")
            self.ClassiMasker = cv2.imread(f"ClassiMask/{CodeSetup}_{ClientName}_{model_number}.png")
            self.SubClassiMask = cv2.imread(f"ClassiMask/{CodeSetup}_{ClientName}_{model_number}_sub.png")
        except:
            print("Classi Masking File Is Nothing")

        testimg = cv2.imread("test.jpg")
        self.inspection(testimg, True)

        logger.info("[Notice] Model reLoading Complete")
        print("[★] 새로운 모델 준비 완료 ! ")

        self.ReloadSignal = True # 모델 재로딩 완료 신호

    # OH
    def Det_load_models(self, model_number):
        try:
            model = tf.Graph()
            model_path = f"models/frozen_inference_graph.pb"
            label_path = f"models/classes.pbtxt"
            with model.as_default():
                # initialize the graph definition
                graphDef = tf.GraphDef()

                # load the graph from disk
                with tf.gfile.GFile(model_path, "rb") as f:
                    serializedGraph = f.read()
                    graphDef.ParseFromString(serializedGraph)
                    tf.import_graph_def(graphDef, name="")

            labelMap = label_map_util.load_labelmap(label_path)
            categories = label_map_util.convert_label_map_to_categories(
                labelMap, max_num_classes= 8 ,
                use_display_name=True)
            self.categoryIdx = label_map_util.create_category_index(categories)

            # create a session to perform inference
            with model.as_default():
                self.sess = tf.Session(graph=model)

            # grab a reference to the input image tensor and the boxes
            # tensor
            self.imageTensor = model.get_tensor_by_name("image_tensor:0")
            self.boxesTensor = model.get_tensor_by_name("detection_boxes:0")

            # for each bounding box we would like to know the score
            # (i.e., probability) and class label
            self.scoresTensor = model.get_tensor_by_name("detection_scores:0")
            self.classesTensor = model.get_tensor_by_name("detection_classes:0")
            self.numDetections = model.get_tensor_by_name("num_detections:0")

            logger.info("[Notice] Detection Model Loading Complete")
            image = cv2.imread("Det_test.jpg")
            logger.info("[Notice] Detection Model Test Start")
            resultLabel, Checkindex, _ = self.Det_inspection(image, self.EngraveCheckLimit[Socket_main.nowModel], 1)
            print(resultLabel, Checkindex)
            logger.info("[Notice] Detection Model Test Complete")
        except:
            print(traceback.format_exc())
            logger.info(f'[Error] ObjectDetection Model Load Error \n{traceback.format_exc()}')

    # OH
    def Det_inspection(self, image, boxlimit, frame):
        try:
            self.DetClassiImage = []
            min_confidence = 0.7
            (reY, reX, reH, reW, ModelType) = self.axisC[Socket_main.nowModel]

            cropimg = image[reY:reY+reH, reX:reX+reW]
            cv2.imwrite('Detection.jpg', cropimg)
            output = cropimg.copy()
            cropimg = cv2.imread('Detection.jpg')
            # cropimg = cv2.resize(cropimg, dsize = (1024, 1024), interpolation = cv2.INTER_AREA)
            
            classImage = cropimg.copy()
            (H, W) = cropimg.shape[:2]
            image_center = (W // 2, H // 2)
            dummy_image = np.zeros( (H, W), dtype = np.uint8)

            cropimg = cv2.cvtColor(cropimg, cv2.COLOR_BGR2RGB)
            cropimg = np.expand_dims(cropimg, axis=0)

            (boxes, scores, labels, N) = self.sess.run(
            [self.boxesTensor, self.scoresTensor, self.classesTensor, self.numDetections],
            feed_dict={self.imageTensor: cropimg})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)
            
            indexCount = 0 # oh
            for (box, score, label) in zip(boxes, scores, labels):
                # if the predicted probability is less than the minimum
                # confidence, ignore it
                if score < min_confidence:
                    continue

                # scale the bounding box from the range [0, 1] to [W, H]
                (startY, startX, endY, endX) = box
                startX = int(startX * W)
                startY = int(startY * H)
                endX = int(endX * W)
                endY = int(endY * H)

                centerX = int(startX/2 + endX/2)
                centerY = int(startY/2 + endY/2)

                # draw the prediction on the output image
                label = self.categoryIdx[label]
                idx = int(label["id"])
                showlabel = "{}:{:.2f}%".format(label["name"], score*100)

                # 분류를 위한 회전 정렬
                x, y = startX, startY
                w, h = endX - startX, endY - startY
                y2 = y + h
                x2 = x + w

                x = x - 25
                y = y - 25
                x2 = x2 + 25
                y2 = y2 + 25

                midX = startX + int((w)/2)
                midY = startY + int((h)/2)
                degree = math.degrees(math.atan2(-midY + image_center[1], midX - image_center[0]))  # 이미지의 중심을 사용하여 각도 계산

                if x < 0: x = 0
                if y < 0: y = 0

                if y2 > classImage.shape[0] :
                    y2 = classImage.shape[0]
                if x2 > classImage.shape[1]:
                    x2 = classImage.shape[1]
                
                crop = classImage[y : y2, x : x2]
                crop = imutils.rotate(crop, -degree)
                crop = cv2.resize(crop, dsize = (224, 224), interpolation = cv2.INTER_AREA)
                # self.DetClassiImage.append(crop)
                # cv2.imwrite('crop.jpg', crop)

                now = datetime.now()
                formatted_date = now.strftime('%Y_%m_%d')
                formatted_time = now.strftime('%H_%M_%S')
                mic = str(now.microsecond).zfill(6)

                path = f"det_classi/{formatted_date}/{Socket_main.nowModel}"
                inputFileName = f'{formatted_date}_{formatted_time}{mic}.jpg'
                MQ.put((path, inputFileName, crop))

                cv2.rectangle(dummy_image, (startX, startY), (endX, endY), (255, 255, 255), -1)
                indexCount += 1 # oh

            # _, dummy_image = cv2.threshold(dummy_image, 127, 255, cv2.THRESH_BINARY)
            # conts = cv2.findContours(dummy_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # conts = imutils.grab_contours(conts)

            # indexCount = 0

            # for idx, cont in enumerate(conts):
            # 	indexCount += 1
            
            if indexCount == boxlimit :  # 박스 수 4개 아니면 NG
                # 분류 모델
                # for i in range(indexCount):
                #     Classilabel, Classiamount = self.DetClassi_inspection(self.DetClassiImage[i])
                #     if 'NG' in Classilabel and Classiamount > 90:
                #         print(" [★] 각인 분류 모델 불량 판정! ")
                #         path = f"DetClassiNg/{Classilabel}"
                #         self.NgDetClassiCount += 1
                #         inputNum = "%05d" % self.NgDetClassiCount
                #         MQ.put((path, "SaveImg_{}.jpg".format(inputNum), self.DetClassiImage[i]))
                #         return "ClassiNG", indexCount
                if self.ED_ok_send_count < 1 :
                    msg = f"{ClientSetup}OKFRAME0OK"
                    Socket_main.ClientSocketSend(msg)
                    pickleData = pickle.dumps(output)
                    Socket_main.client_socket.send(str(len(pickleData)).ljust(20).encode())
                    Socket_main.client_socket.send(pickleData)
                    self.ED_ok_send_count += 1
                return "OK", indexCount, output
            else :
                if self.ED_ng_send_count < 2 :
                    msg = f"{ClientSetup}NGFRAME0ED"
                    Socket_main.ClientSocketSend(msg)
                    pickleData = pickle.dumps(output)
                    Socket_main.client_socket.send(str(len(pickleData)).ljust(20).encode())
                    Socket_main.client_socket.send(pickleData)
                    self.ED_ng_send_count += 1
                return "NG", indexCount, output

        except:
            print('에러 발생으로 강제 불량 판정', traceback.format_exc()) # OH
            logger.info(f'[Error] Detecton Error \n{traceback.format_exc()}')
            return "NG", 99, image
    
    def DetClassi_load_models(self, model_number):
        ClassimodelPath = f"models/{model_number}/Classi/model.hdf5"
        ClassipicklePath = f"models/{model_number}/Classi/model.pickle"
        self.Classimodel = load_model(ClassimodelPath)
        self.Classilb = pickle.loads(open(ClassipicklePath, 'rb').read())
        print("[★] 디텍션 - 분류 모델 로드 완료")
        testimg = cv2.imread("test.jpg")
        self.DetClassi_inspection(testimg)
        logger.info("[Notice] Detection - Classification Model Loading Complete")

    def DetClassi_inspection(self, img):
        try:
            now = datetime.now()
            nowDatetime = now.strftime("%H-%M-%S")
            nowDatetime = nowDatetime.split('-')
            filename = nowDatetime[0]+nowDatetime[1]+nowDatetime[2]
            img = cv2.resize(img, (224, 224))
            output = img.copy()
            img = img.astype("float") / 255.0
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)

            proba = self.Classimodel.predict(img)[0]
            idx = np.argmax(proba)
            label = self.Classilb.classes_[idx]
            amount = (proba[idx]) * 100
            
            result = f'{label}_{str(amount)}'
            self.DetClassiResult.append(result)
            print("*********", label, amount)
            foldername = f'Det_Test/{label}'
            if not (os.path.isdir(foldername)):
                os.makedirs(os.path.join(foldername))
            cv2.imwrite(f'{foldername}/{filename}.jpg', output)
            return label, amount

        except:
            print(traceback.format_exc())
            return "OK", 0.9

    def checkValueDictUpdate(self, dictdata):
        for name, value in dictdata:
            self.checkValueDict[name] = value

    def Definition_Inspection(self, Image):
        startTime = time.time()
        def Sobel(img):
            img_sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

            img_sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

            img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)

            return img_sobel

        def get_Definition(img_sobel):
            sum_data = 0

            for y in range(0, len(img_sobel), 1):
                for x in range(0, len(img_sobel[y]), 1):
                    sum_data += (int(img_sobel[y, x]) ** 2)
            sum_data = sum_data / (len(img_sobel) * len(img_sobel[y]))

            return int(sum_data)

        img_gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

        img_data = Sobel(img_gray)
        Definition = get_Definition(img_data)
        print(f'선명도 체크 결과수치 - {Definition}')
        logger.info(f'Definition Inspection Result Score - {Definition}')
        logger.info(f'Definition Inspection Time - {time.time() - startTime}')

        return Definition

    def ClassiInspection(self, oriOutput, images, FrameCount):
        sub_image = images.copy()
        self.allDone = False
        # startTime = time.time()
        continuityDetectImageUpdateTriger = False
        criticalDetectImageUpdateTriger = False
        BadDataCheck = False
        ReturnBadCounting = []
        try:
            if FrameCount == 1:
                self.resultOKImageData = oriOutput.copy()

            if FrameCount in [5,10,15]:
                #외륜 104, 내륜 111 각인 검증
                if (CodeSetup == "CUP" and ClientName == "CLIENT0" and (Socket_main.nowModel in EFFI.Det_ModelSetup['CUP'])) or (CodeSetup == "CONE" and ClientName == "CLIENT1" and (Socket_main.nowModel in EFFI.Det_ModelSetup['CONE'])) or (LINE == 'SS11' and CodeSetup == "CUP" and ClientName == "CLIENT1" and (Socket_main.nowModel == "MODEL7" or Socket_main.nowModel == "MODEL8")):
                    if CodeSetup == "CONE" and ClientName == "CLIENT1" : # 내륜 111번 각인 검사 서브 마스킹으로 대체and (Socket_main.nowModel in EFFI.Det_ModelSetup['CUP'])
                        det_ori_image = oriOutput.copy()
                        det_image = cv2.bitwise_and(det_ori_image, self.SubClassiMask)
                    else :
                        det_image = images
                        
                    if EFFI.Det_Running == True:
                        # SS11라인 101 식별 각인 추가
                        if LINE == 'SS11' and CodeSetup == "CUP" and ClientName == "CLIENT1" and (Socket_main.nowModel == "MODEL7" or Socket_main.nowModel == "MODEL8"):
                            resultEngraveLabel, resultEngraveCount, det_output_image = self.Det_inspection(det_image, EFFI.EngraveCheckLimit_101[Socket_main.nowModel], FrameCount)
                            self.Det_resultLabel[1] = resultEngraveCount

                            #각인 갯수 검사결과 갱신
                            if EFFI.EngraveCheckLimit_101[Socket_main.nowModel] == resultEngraveCount:
                                print(f'[Engrave Detect Count] - Count Check OK / Limit : {EFFI.EngraveCheckLimit_101[Socket_main.nowModel]}, Result : {resultEngraveCount}')
                            else:
                                print(f'[Engrave Detect Count] - Count Check NG / Limit : {EFFI.EngraveCheckLimit_101[Socket_main.nowModel]}, Result : {resultEngraveCount}')

                            #각인 라벨 검사결과 갱신
                            if resultEngraveLabel == 'OK':
                                print(f'[Engrave Detect Label] - Label Check OK')
                                self.Det_resultLabel[0] = resultEngraveLabel
                            else:
                                if self.Det_resultLabel[0] == 'OK' :
                                    pass
                                else : 
                                    print(f'[Engrave Detect Label] - Label Check NG - Result : {resultEngraveLabel}')
                                    if self.Det_resultLabel[0] != 'OK':
                                        self.Det_resultLabel[0] = resultEngraveLabel

                            if ('NG' in resultEngraveLabel) or (EFFI.EngraveCheckLimit_101[Socket_main.nowModel] != resultEngraveCount):
                                path = f"DetectionNg/{resultEngraveCount}"
                                now = datetime.now()
                                year = now.year
                                month = now.month
                                day = now.day
                                hour = now.hour
                                minute = now.minute
                                second = now.second
                                mic = now.microsecond
                                inputFileName = f'{Socket_main.nowModel}_{year}_{month}_{day}_{hour}_{minute}_{second}_{mic}_{resultEngraveCount}Count.jpg'
                                # DetLoadImg = cv2.resize(images, dsize = (1024, 1024), interpolation = cv2.INTER_AREA)
                                MQ.put((path, "SaveImg_{}.jpg".format(inputFileName), det_output_image))

                        else :
                            resultEngraveLabel, resultEngraveCount, det_output_image = self.Det_inspection(det_image, EFFI.EngraveCheckLimit[Socket_main.nowModel], FrameCount)
                            
                            # self.Det_resultLabel[0], self.Det_resultLabel[1]
                            #각인 갯수 검사결과 갱신
                            if EFFI.EngraveCheckLimit[Socket_main.nowModel] == resultEngraveCount:
                                print(f'[Engrave Detect Count] - Count Check OK / Limit : {EFFI.EngraveCheckLimit[Socket_main.nowModel]}, Result : {resultEngraveCount}')
                                self.Det_resultLabel[1] = resultEngraveCount
                            else:
                                print(f'[Engrave Detect Count] - Count Check NG / Limit : {EFFI.EngraveCheckLimit[Socket_main.nowModel]}, Result : {resultEngraveCount}')
                                if self.Det_resultLabel[1] != EFFI.EngraveCheckLimit[Socket_main.nowModel]:
                                    self.Det_resultLabel[1] = resultEngraveCount

                            #각인 라벨 검사결과 갱신
                            if resultEngraveLabel == 'OK':
                                print(f'[Engrave Detect Label] - Label Check OK')
                                self.Det_resultLabel[0] = resultEngraveLabel
                            else:
                                if self.Det_resultLabel[0] == 'OK' :
                                    pass
                                else : 
                                    print(f'[Engrave Detect Label] - Label Check NG - Result : {resultEngraveLabel}')
                                    if self.Det_resultLabel[0] != 'OK':
                                        self.Det_resultLabel[0] = resultEngraveLabel

                            if ('NG' in resultEngraveLabel) or (EFFI.EngraveCheckLimit[Socket_main.nowModel] != resultEngraveCount):
                                path = f"DetectionNg/{resultEngraveCount}"
                                now = datetime.now()
                                year = now.year
                                month = now.month
                                day = now.day
                                hour = now.hour
                                minute = now.minute
                                second = now.second
                                mic = now.microsecond
                                inputFileName = f'{Socket_main.nowModel}_{year}_{month}_{day}_{hour}_{minute}_{second}_{mic}_{resultEngraveCount}Count.jpg'
                                # DetLoadImg = cv2.resize(images, dsize = (1024, 1024), interpolation = cv2.INTER_AREA)
                                MQ.put((path, "SaveImg_{}.jpg".format(inputFileName), det_output_image))

            for i in range(self.PartCounting): # oh
                if i == 3 and (CodeSetup == "CONE" and ClientName == "CLIENT1") : # 111 이종 혼입 검사
                    if FrameCount in [5, 10, 15]: # 5, 10, 15 프레임만 검사
                        print(f"[★] 이종 혼입 검사 시작 - 검사 프레임 {FrameCount}")
                        pass
                    else :
                        continue
                    
                if i == 3 and (CodeSetup == "CONE" and ClientName == "CLIENT0") : # 114 이종 혼입 검사
                    if FrameCount in [5, 10, 15]: # 5, 10, 15 프레임만 검사
                        print(f"[★] 이종 혼입 검사 시작 - 검사 프레임 {FrameCount}")
                        pass
                    else :
                        continue
                
                if i == 2 and (CodeSetup == "CUP" and ClientName == "CLIENT0") : # 104 이종 혼입 검사
                    if FrameCount in [5, 10, 15]: # 5, 10, 15 프레임만 검사
                        print(f"[★] 이종 혼입 검사 시작 - 검사 프레임 {FrameCount}")
                        pass
                    else :
                        continue

                if i == 1 and (CodeSetup == "CONE" and ClientName == "CLIENT5") : # 116 이종 혼입 검사
                    if FrameCount in [2, 4, 6, 8, 10, 12, 14 ,16 ,18, 20, 22, 24, 26, 28, 30]: # 15개 프레임만 검사
                        print(f"[★] 이종 혼입 검사 시작 - 검사 프레임 {FrameCount}")
                        pass
                    else :
                        continue

                # SS9 궤도 검사 구역 추가된 형번
                if i == 0 and (LINE == 'SS9' and CodeSetup == "CUP" and ClientName == "CLIENT2") :
                    if Socket_main.nowModel == "MODEL4" or Socket_main.nowModel == "MODEL5" or Socket_main.nowModel == "MODEL6" :
                        images = cv2.bitwise_and(images, EFFI.ClassiMasker)
                elif i == 1 and (LINE == 'SS9' and CodeSetup == "CUP" and ClientName == "CLIENT2") :
                    if Socket_main.nowModel == "MODEL4" or Socket_main.nowModel == "MODEL5" or Socket_main.nowModel == "MODEL6" :
                        if FrameCount in [4,8,12,16,20,24,28,32,36,40]: # PART2 10장만 검사 그 외는 패스
                            cv2.imwrite('sub_image.jpg', sub_image)
                            row_image = cv2.imread('sub_image.jpg')
                            images = cv2.copyTo(row_image, EFFI.SubClassiMask, EFFI.SubClassiMask)
                            images = cv2.convertScaleAbs(images, alpha=3, beta=0)
                            print(f"[★] 궤도 하단부 추가 검사 - 검사 프레임 {FrameCount}")
                        else :
                            continue

                rect_image = images[self.cropBox[Socket_main.nowModel][i][0] : self.cropBox[Socket_main.nowModel][i][0] + self.cropBox[Socket_main.nowModel][i][2],
                                    self.cropBox[Socket_main.nowModel][i][1] : self.cropBox[Socket_main.nowModel][i][1] + self.cropBox[Socket_main.nowModel][i][3]].copy()
                
                if len(self.cropBox[Socket_main.nowModel][i]) != 4  :             # 회전 재원이 입력되어 있는 경우만 실행
                    crop_rotate_angle = self.cropBox[Socket_main.nowModel][i][4]  # 이미지 회전 각도
                    coord = self.cropBox[Socket_main.nowModel][i][0:4]
                    matrix = cv2.getRotationMatrix2D((int(coord[3]/2), int(coord[2]/2)), crop_rotate_angle, 1)
                    rect_image = cv2.warpAffine(rect_image, matrix, (coord[3], coord[2]))
                    cv2.imwrite('rotate.jpg', rect_image)         

                ##################################################################################
                if FrameCount == 1 and i == 0:
                    resultLabel, score = self.inspection(rect_image, True)
                else:
                    resultLabel, score = self.inspection(rect_image, False)

                # resultLabel = 'NG'
                # score = 90
                # self.Definition_Result = 999

                if 'MIX' in resultLabel :
                    print(f"[★] 이종 혼입 불량 발생 - 검사 프레임 {FrameCount}")
                else :
                    print(f"Part{i+1} > Label : {resultLabel}, score : {score}")
                
                # 불량 판정
                if 'OK' not in resultLabel:
                    # 불량 이미지 전송
                    size = self.modelResizeDict[Socket_main.nowModel]
                    savedImage = cv2.resize(rect_image, (size[1], size[0]))

                    if self.ng_send_count_list[i] < 2 : # 불량 2장만 전송
                        msg = f"{ClientSetup}NGFRAME{i+1}{resultLabel}"
                        Socket_main.ClientSocketSend(msg)
                        pickleData = pickle.dumps(savedImage)
                        Socket_main.client_socket.send(str(len(pickleData)).ljust(20).encode())
                        Socket_main.client_socket.send(pickleData)
                        print('불량 이미지 전송')
                        self.ng_send_count_list[i] += 1

                    if ODC.NgCaptureCheck == True:
                        now = datetime.now()
                        now = datetime.now()
                        year = str(now.year).zfill(4)
                        month = str(now.month).zfill(2)
                        day = str(now.day).zfill(2)
                        hour = str(now.hour).zfill(2)
                        minute = str(now.minute).zfill(2)
                        second = str(now.second).zfill(2)
                        mic = str(now.microsecond).zfill(6)
                        path = f"NgCapture/{year}_{month}_{day}/{Socket_main.nowModel}/{self.SessionCheckList[i]}/{resultLabel}"
                        inputFileName = f'{year}_{month}_{day}_{hour}_{minute}{second}{mic}.jpg'
                        MQ.put((path, inputFileName, savedImage))
                
                # 양품 판정
                else :
                    if self.ok_send_count_list[i] < 1 : # 양품 1장만 전송
                        # 양품 이미지 전송
                        size = self.modelResizeDict[Socket_main.nowModel]
                        savedImage = cv2.resize(rect_image, (size[1], size[0]))

                        msg = f"{ClientSetup}OKFRAME{i+1}{resultLabel}"
                        Socket_main.ClientSocketSend(msg)
                        pickleData = pickle.dumps(savedImage)
                        Socket_main.client_socket.send(str(len(pickleData)).ljust(20).encode())
                        Socket_main.client_socket.send(pickleData)
                        print('양품 이미지 전송')
                        self.ok_send_count_list[i] += 1

                if CTH.CaptureMode_Process == True:
                    size = self.modelResizeDict[Socket_main.nowModel]
                    savedImage = cv2.resize(rect_image, (size[1], size[0]))
                    now = datetime.now()
                    year = str(now.year).zfill(4)
                    month = str(now.month).zfill(2)
                    day = str(now.day).zfill(2)
                    hour = str(now.hour).zfill(2)
                    minute = str(now.minute).zfill(2)
                    second = str(now.second).zfill(2)
                    mic = str(now.microsecond).zfill(6)
                    path = f"Capture_Train"
                    inputFileName = f'{year}_{month}_{day}_{hour}_{minute}{second}{mic}.jpg'
                    MQ.put((path, inputFileName, savedImage))

                # 중대불량 체크
                for CheckLabel in EFFI.CriticalProductSetup[ClientName]:
                    if (CheckLabel in resultLabel and score > self.checkValueDict[CheckLabel][0]):
                        print('\n\n 중 대 불 량 발 생 \n\n')
                        cv2.rectangle(oriOutput, (self.cropBox[Socket_main.nowModel][i][1], self.cropBox[Socket_main.nowModel][i][0]), (self.cropBox[Socket_main.nowModel][i][1]+self.cropBox[Socket_main.nowModel][i][3], self.cropBox[Socket_main.nowModel][i][0]+self.cropBox[Socket_main.nowModel][i][2]), (0,0,255), 5)
                        EFFI.CriticalProductResultDict[CheckLabel][1] += 1
                        EFFI.CriticalProductResultDict[CheckLabel][3] += 1

                        # if EFFI.CriticalProductResultDict[CheckLabel][1] >= self.checkValueDict[CheckLabel][1]:
                        if EFFI.CriticalProductResultDict[CheckLabel][1] >= EFFI.CriticalProductResultDict[CheckLabel][2]:
                            EFFI.CriticalProductResultDict[CheckLabel][2] = EFFI.CriticalProductResultDict[CheckLabel][1]
                            # EFFI.CriticalProductResultDict[CheckLabel][0] = True
                        
                        criticalDetectImageUpdateTriger = True
                        BadDataCheck = True
                    else:
                        EFFI.CriticalProductResultDict[CheckLabel][1] = 0
                        pass

                # 일반 파트별 검사 체크
                print(f'{resultLabel} , {score} > {self.checkValueDict[f"PART{i+1}"][0]}')
                if ("NG" in resultLabel and score > self.checkValueDict[f"PART{i+1}"][0]):
                    # print(f"Part{i+1} Result Check {self.SessionCheckList[i]} : {resultLabel}")
                    cv2.rectangle(
                        oriOutput, (self.cropBox[Socket_main.nowModel][i][1], self.cropBox[Socket_main.nowModel][i][0]), (self.cropBox[Socket_main.nowModel][i][1] + self.cropBox[Socket_main.nowModel][i][3], self.cropBox[Socket_main.nowModel][i][0] + self.cropBox[Socket_main.nowModel][i][2]), (0, 0, 255), 5,
                    )

                    self.BadCheckCount[i] += 1
                    if self.BadCheckCount[i] >= self.checkValueDict[f"PART{i+1}"][1]: 
                        print(f"Continuity Bad Detect Check - Part{i+1}")
                        self.continuityDetectState = True
                        continuityDetectImageUpdateTriger = True

                    self.resultImageData[i] = oriOutput.copy()
                    ReturnBadCounting.append(i)
                    BadDataCheck = True

                else:
                    self.BadCheckCount[i] = 0
                    print(f"Part Result Check {i+1} : OK")
                    # cv2.rectangle(oriOutput, (self.cropBox[Socket_main.nowModel][i][1], self.cropBox[Socket_main.nowModel][i][0]), (self.cropBox[Socket_main.nowModel][i][1]+self.cropBox[Socket_main.nowModel][i][3], self.cropBox[Socket_main.nowModel][i][0]+self.cropBox[Socket_main.nowModel][i][2]), (0,255,0), 10)

            if continuityDetectImageUpdateTriger == True:
                self.continuityDetectImage = oriOutput.copy()

            #중대불량 불량이미지 백업
            if criticalDetectImageUpdateTriger == True:
                for CriticalLabel in EFFI.CriticalProductSetup[ClientName]:
                    if EFFI.CriticalProductResultDict[CriticalLabel][1] >= EFFI.checkValueDict[CriticalLabel][1]:
                        EFFI.CriticalProductResultDict[CriticalLabel][0] = False
                        EFFI.CriticalProductImageDict[CriticalLabel] = oriOutput.copy()
                    if EFFI.CriticalProductResultDict[CriticalLabel][3] >= EFFI.checkValueDict[CriticalLabel][2]:
                        EFFI.CriticalProductImageDict[CriticalLabel] = oriOutput.copy()

            #최종 불량이미지 백업
            if BadDataCheck == True:
                self.LastBadImage = oriOutput.copy()

            # print("[INFO] Elapesed time : ", time.time() - startTime)

            return ReturnBadCounting
        except:
            print(f'[Error] Inspection Process Error - {traceback.format_exc()}')
            logger.info(f'[Error] Inspection Process Error - {traceback.format_exc()}')
            return []

    def inspection(self, img, FirstCheck):
        try:
            size = self.modelResizeDict[Socket_main.nowModel]
            img = cv2.resize(img, (size[1], size[0]))

            cv2.imwrite('Inspection.jpg', img)
            img = cv2.imread('Inspection.jpg')

            # if FirstCheck == True:
            #     self.Definition_Result = self.Definition_Inspection(img)

            # print(img.shape)
            # output = img.copy()
            img = img.astype("float") / 255.0
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)

            # OH
            # tf2 MODEL
            # proba = self.model.predict_on_batch(img)[0]

            # tf1 MODEL
            # proba = self.model.predict(img)[0]
            # with self.graph_classi.as_default():

            proba = self.model.predict(img)[0]
            idx = np.argmax(proba)
            label = self.lb.classes_[idx]
            amount = (proba[idx]) * 100

            # if "NG" in label and amount < 0.85:
            # 	label = "OK"

            # label = 'OK'
            # amount = 99

            return label, amount
        except:
            print('에러 발생으로 강제 불량 판정', traceback.format_exc()) # OH
            logger.info(f'[Error] Classification Error \n{traceback.format_exc()}')
            return "NG", 99


class ObjectDetectImg:
    def __init__(self):
        # threading.Thread.__init__(self)
        self.min_confidence = 0.3
        self.inspectionCheck = False
        self.ModelLoadComp = False
        self.badSearchCheck = []

        self.NgCaptureCheck = False


    def inspectionIMG(self):

        imageInspectionCount = 0
        # resultNgCheck = 0
        startTime = time.time()

        logger.info("[Notice] Inspection Start")
        print("검사 진행중")

        EFFI.resultPartData = [0] * EFFI.PartCounting
        DataEmptyCount = 0

        while True:
            try:
                if not CTH.Qimage.empty():
                    startTime_OneImageCycle = time.time()
                    DataEmptyCount = 0
                    images = CTH.Qimage.get()
                    imageInspectionCount += 1

                    output = images.copy()

                    # print(f'Check1 - {time.time() - startTime_OneImageCycle}')
                    
                    # classification inspection add
                    # ******************************************************************************************************************************************************
                    if imageInspectionCount == 1: # 첫장
                        # 9라인 외륜 ST-E508422 궤도 하단부 검사 구역 추가
                        if LINE == 'SS9' and CodeSetup == "CUP" and ClientName == "CLIENT2" :
                            if Socket_main.nowModel == "MODEL4" or Socket_main.nowModel == "MODEL5" or Socket_main.nowModel == "MODEL6" :
                                # 첫 프레임 마스킹 확인하기 위해 저장
                                maskedImage = cv2.bitwise_and(images, EFFI.ClassiMasker)
                                cv2.imwrite("Masking.jpg", maskedImage)
                                submaskedImage = cv2.bitwise_and(images, EFFI.SubClassiMask)
                                cv2.imwrite("SubMasking.jpg", submaskedImage)
                                ClassiResult = EFFI.ClassiInspection(output, images, imageInspectionCount) # 마스킹 ClassiInspection 함수 내에서 실행
                            else :
                                classiImage = cv2.bitwise_and(images, EFFI.ClassiMasker)
                                cv2.imwrite("Masking.jpg", classiImage)
                                ClassiResult = EFFI.ClassiInspection(output, classiImage, imageInspectionCount)
                        else : 
                            classiImage = cv2.bitwise_and(images, EFFI.ClassiMasker)
                            cv2.imwrite("Masking.jpg", classiImage)
                            ClassiResult = EFFI.ClassiInspection(output, classiImage, imageInspectionCount)
                    else: 
                        if LINE == 'SS9' and CodeSetup == "CUP" and ClientName == "CLIENT2":
                            if Socket_main.nowModel == "MODEL4" or Socket_main.nowModel == "MODEL5" or Socket_main.nowModel == "MODEL6" :
                                ClassiResult = EFFI.ClassiInspection(output, images, imageInspectionCount)
                            else : 
                                classiImage = cv2.bitwise_and(images, EFFI.ClassiMasker)
                                ClassiResult = EFFI.ClassiInspection(output, classiImage, imageInspectionCount)
                        else : 
                            classiImage = cv2.bitwise_and(images, EFFI.ClassiMasker)
                            ClassiResult = EFFI.ClassiInspection(output, classiImage, imageInspectionCount)

                    # print(f'Check2 - {time.time() - startTime_OneImageCycle}')

                    for i in ClassiResult:
                        checkIndex = int(i)
                        EFFI.resultPartData[checkIndex] += 1

                    print(f"[Notice] Inspection Complete - {imageInspectionCount} / {time.time() - startTime_OneImageCycle}")

                    if ClientName == 'CLIENT1' or ClientName == 'CLIENT0':
                        checkFrame = 21
                    else:
                        checkFrame = 43

                    if imageInspectionCount >= checkFrame:
                        print("InsepctionIMG inspection Count OVER")
                        print("InspectionIMG Thread Closed")
                        logger.info("[Notice] Inspection Thread Closed")
                        self.inspectionCheck = True
                        break

                else:
                    print("data empty check")
                    if Socket_main.resultSession == True:
                        print("InspectionIMG Thread Closed")
                        logger.info("[Notice] Inspection Thread Closed")
                        self.inspectionCheck = True
                        break

                    time.sleep(0.1)

                    if time.time() - startTime > 8:
                        print("InspectionIMG TIME OUT Thread Closed")
                        logger.info("[Notice] Inspection TIME OUT Thread Closed")
                        break

            except:
                logger.info(f"Warning : inspection work error \n {traceback.format_exc()}")


def SaveImages(MQ):
    while True:
        if not MQ.empty():
            try:
                data = MQ.get()
                foldername = data[0]
                filename = data[1]

                if not (os.path.isdir(foldername)):
                    os.makedirs(os.path.join(foldername))

                cv2.imwrite(foldername + "/" + filename, data[2])

            except:
                pass


if __name__ == "__main__":
    Socket_main = SocketCommunication()

    ODC = ObjectDetectImg()
    EFFI = EfficientNetIMG()

    CTH = cameraRTSP()
    CTH.daemon = True
    CTH.start()

    MQ = MQueue()
    proc = Process(target=SaveImages, args=(MQ,))
    proc.daemon = True
    proc.start()

    threading.Thread(target=CTH.remove_forder, daemon=True).start() # OH
    
    while True:
        try:
            Socket_main.connectTry()
            Socket_main.run()
            time.sleep(1)
        except Exception as ex:
            print(ex)  # [Errno 111] Connection refused
            time.sleep(1)
            print("Notice : [Socket Disconnected. Connect Retry]\n")
            logger.info(f"Warning : 소켓 프로그램 종료, 재실행중 - {ex}")
