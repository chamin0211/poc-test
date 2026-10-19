import h2o

# init()은 서버를 새로 띄우려 하니 connect()로 기존 서버에만 붙는다
h2o.connect(ip="127.0.0.1", port=54321)
print("cloud_size:", h2o.cluster().cloud_size)

fr = h2o.H2OFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
print(fr.describe())
print("check ok")
