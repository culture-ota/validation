### 실험 결과 데이터와 실험 코드는 Repository 내 각 폴더로 정리
### 현재는 공전원 서버 내 로컬 접속 환경만 정리되어 있음. 자체 로컬 서버에서의 동작 가이드는 별도 작성 예정

#공전원 3번 서버로 SSH 접속 - IP는 연구실 내부에만 공유되어 있음
#계정 seyeolyang 비밀번호 aril0246

###검증 1. Cultural Token
docker stop c3476e27ebcf
docker start c3476e27ebcf
#(docker stop 및 start는 최초시에만, 이후 창을 열어서 docker 접속시 매 창에 대해 아래 4개 명령어 모두 실행)

docker exec -it -e DISPLAY=$DISPLAY c3476e27ebcf /bin/bash
export DISPLAY=localhost:12.0 
conda activate e2map 
source /home/e2map/devel/setup.bash

#이후 docker 내부에 진입해서 xclock 명령시 시계가 안뜬다면 X11 forwarding 실패로, 다시 exit를 통해 docker 외부에서의 DISPLAY 값으로 일치시켜 주어야 함
#echo $DISPLAY 시 확인되는 localhost:# 값으로 export DISPLAY의 localhost:# 값으로 변경 해야함. 이후 docker 내부에서 xclock 시 동작해야 함

#1번 윈도우
roscore ([roscore 실행이 안될때 - RLException: Unable to contact my own server at [http://ubuntu:43391/])
sudo nano /etc/hosts
127.0.0.1       localhost
127.0.0.1       ubuntu        #이걸 추가
192.168.0.84    ubuntu

#2번 윈도우
roslaunch clean_e2map ikea_1.launch #(ikea_2 ~ 5 동일)

#3번 윈도우
roslaunch irobot spawn_Robot1.launch #(ikea_2 ~ 5 동일)

#4번 윈도우
roslaunch rviz_map rviz_map.launch

#5번 윈도우 (dirt map)
python /home/e2map/clean_map/dirt_map.py #해당 코드 실행 후 rivz 상 /dirt_map_image의 Image 토픽 등록 시 실시간 잔여 오염도 확인 가능

#6번 윈도우 (최종 실행 코드)
python /home/e2map/clean_map/move_final.py

#청소 결과 및 culture persona input시 LLAMA3.1 Reasoning (cultural.txt 가 output 파일)
python /home/e2map/clean_map/llama3_input_2.py

#Culture token 생성 (cultural_token.txt 저장)
python /home/e2map/clean_map/llama3_input_1.py

>token 기반 local policy 생성
#Path planning 코드 (전체 SLAM map이 완성되있는 상태에서 해야 path.txt 추출가능)
python /home/e2map/clean_map/path_planing_1.py

>Obstacle avoid 코드 
>LoRA Fine-tuning


###검증 2. OTA Update (ns-3 simulator)

##Global Policy update (호스트 서버 → 도커 내부)

#1번 윈도우 (도커 외부, 호스트 서버)
conda activate python3.10
cd /data/seyeolyang/uptane
python -i manufacturer.py
#같은 윈도우의 인터프리터 내부에서 아래 명령 수행
firmware_fname = filepath_in_repo = 'global_policy_1.yaml'   # 같은 폴더내에 위치하고 있는 글로벌 정책 파일
di.add_target_to_imagerepo(firmware_fname, filepath_in_repo) # Image repo 폴더에 Fireware로 등록
di.write_to_live()                                           # 검증용 metadate 작성
vin='vacuum_1'; ecu_serial='global_1'                        # Target 정보 생성
dd.add_target_to_director(firmware_fname, filepath_in_repo, vin, ecu_serial) # Director repo로 Fireware 지정
dd.write_to_live(vin_to_update=vin)                                          # 해당 ECU에 실제 배포

#2번 윈도우 (도커 내부)
cd /home/uptane
python
#같은 윈도우의 인터프리터 내부에서 아래 명령 수행
import ECU as ecu
ecu.clean_slate()   #ECU Initialize 

#3번 윈도우 (도커 내부)
cd /home/uptane
python
#같은 윈도우의 인터프리터 내부에서 아래 명령 수행
import primary as pr
pr.clean_slate()    #Primary Initialize 
#다시 2번 윈도우에서
ecu.update_cycle()  #OTA update 수행 (제조사 → ECU)
#다시 3번 윈도우에서
pr.update_cycle()   #OTA update 수행 (ECU → Primary) 최종 Primary가 업데이트 되는것이므로 updated 창이 떠야 함.
ctrl+D 로 나와서 같은 폴더인 /home/uptane/global_1/global_policy_1.yaml 이 전송된 것 확인

##Local Policy update (도커 내부 → 호스트 서버)

#1번 윈도우 (도커 내부)
cd /home/uptane
python -i ecu_server.py
firmware_fname = filepath_in_repo = 'delta_cultural_token_1.txt'
di.add_target_to_imagerepo(firmware_fname, filepath_in_repo) 
di.write_to_live()                                          
vin='vacuum_2'; ecu_serial='local_1'                        
dd.add_target_to_director(firmware_fname, filepath_in_repo, vin, ecu_serial)
dd.write_to_live(vin_to_update=vin)                                      

#2번 윈도우 (도커 외부, 제조사 서버)
conda activate python3.10
cd /data/seyeolyang/uptane
python
#같은 윈도우의 인터프리터 내부에서 아래 명령 수행
import manufacturer_server as ms
ms.clean_slate()

#3번 윈도우 (도커 외부, 제조사 스토리지)
cd /home/uptane
python
import manufacturer_tokenstorage as mt
mt.clean_slate()
#다시 2번 윈도우에서
ms.update_cycle()  
#다시 3번 윈도우에서
mt.update_cycle()    # UPDATED 뜨면서 token값 출력 및 저장되면 성공

#ns-3 simulator
#docker 내부에서 아래 폴더에 접속 
cd home/ns-3-allinone/ns-3-dev/ 
#폴더에서 아래 명령어 실행 (Point-to-point UDP test, 4개 network type 동시 전송됨)
./ns3 run token_full_update.cc
./ns3 run token_delta_update.cc
./ns3 run model_update.cc

#시각화 (netanim 기반) 위 코드 실행 시 각각의 xml 파일이 출력됨. xml 파일명을 직접 아래 디렉토리 내에서 이동하여 실행 필요
cd /home/ns-3-allinone/netanim/build/bin
./netanim ~/ns-3-allinone/ns-3-dev/multi-link.xml



