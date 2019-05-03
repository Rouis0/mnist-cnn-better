# mnist-cnn-better
mnist手写数字识别卷积神经网络优化，识别精度99.7%

## 文件结构
```
- src
    - Model.py（模型结构）
    - ModelUtil.py（一些对模型的操作函数）

- logs（TensorBoard记录）

- model（保存的模型）
    - model.png
    - my_model.h5
    - file.json

- train.py（训练模型）
- serve.py（服务器监听）
```

## 安装说明
1. 自定义Model.py的模型结构，train.py的训练参数
2. 运行train.py文件进行模型训练
3. 修改server.py文件的域名以及监听端口
4. 运行server.py文件进行端口监听
5. 在应用中调用接口，附上[web应用代码][1]，以及[博客][2]

[1]: https://github.com/Rouis0/mnist-web
[2]: https://blog.yube.vip/archives/106/