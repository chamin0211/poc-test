#!/bin/bash
# 유령 노드(.254)가 클러스터에 끼지 않게 멤버를 자기 자신 하나로 고정해서 띄운다
pkill -f h2o.jar 2>/dev/null
echo "127.0.0.1:54321" > /tmp/flatfile.txt
JAR=$(python -c "import h2o,os;print(os.path.join(os.path.dirname(h2o.__file__),'backend/bin/h2o.jar'))")
nohup java -Xmx4G -jar "$JAR" -ip 127.0.0.1 -web_ip 127.0.0.1 \
  -baseport 54321 -flatfile /tmp/flatfile.txt -nthreads 2 \
  -name poc_local > /tmp/h2o_poc.log 2>&1 &
sleep 15
curl -s -m 10 http://127.0.0.1:54321/3/Cloud | grep -o '"cloud_size":[0-9]*'
