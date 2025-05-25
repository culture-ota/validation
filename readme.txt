#최종 실험 결과 데이터 및 실험 대상 코드는 별도로 폴더 정리함
#현재는 공전원 서버 내 로컬 파일 접속 환경만 정리되어 있음. 자체 로컬 서버에서도 동작되도록 코드 정리 예정

#공전원 1번 서버로 SSH 접속 - IP는 연구실 내부에만 공유되어 있음
#계정 seyeolyang 비밀번호 aril0246

#검증 1. Cultural Token
docker stop c3476e27ebcf
docker start c3476e27ebcf
(docker stop 및 start는 최초시에만, 이후 창을 열어서 docker 접속시 매 창에 대해 아래 4개 명령어 모두 실행)

docker exec -it -e DISPLAY=$DISPLAY c3476e27ebcf /bin/bash
export DISPLAY=localhost:12.0 
conda activate e2map 
source /home/e2map/devel/setup.bash

이후 docker 내부에서 xclock 시 시계가 안뜬다면 X11 forwarding 실패로, 다시 exit를 통해 docker 외부에서의 DISPLAY 값으로 일치시켜 주어야 함
echo $DISPLAY 시 확인되는 localhost:# 값으로 export DISPLAY의 localhost:# 값으로 변경 해야함

1번창
roscore ([roscore 실행이 안될때 - RLException: Unable to contact my own server at [http://ubuntu:43391/])
sudo nano /etc/hosts
127.0.0.1       localhost
127.0.0.1       ubuntu        #이걸 추가
192.168.0.84    ubuntu

2번창
roslaunch clean_e2map ikea_1.launch (ikea_2 ~ 5 동일)

3번창
roslaunch irobot spawn_iRobot.launch

4번창
roslaunch rviz_map rviz_map.launch

5번창 (실행 코드)
python /home/e2map/clean_map/move_final.py



#검증 2. OTA Update (ns-3 simulator)
docker 내부에서 아래 폴더에 접속 
cd home/ns-3-allinone/ns-3-dev/

./ns3 build

./ns3 run token_full_update.cc

cd /home/ns-3-allinone/netanim/build/bin
./netanim ~/ns-3-allinone/ns-3-dev/multi-link.xml



