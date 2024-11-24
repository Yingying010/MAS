1. URDF problem
![[pic/URDF.png]]

解决方式：
我的urdf文件路径写错啦～！！
改成 data/turtlebot.urdf 就可以了
要注意路径内容 以及 文件夹

2. a_star problem
![[astar.png]]

解决方式：
重新clone了一下代码就好了...

3. invalid buffer size problem
![[pic/invalidBufferSize.png]]

解决方式：
换成ubuntu就好了，可能确实是不兼容mac

4. mac摄像头设备没有no controls 
![[MacCameraNoControlList.png]]

解决方式：
换成windows....
而且mac画质640\*480太糊了

5. windows的ubuntu摄像头打不开
`ls /dev |grep video` 可以查看摄像头是否已经成功链接到ubuntu
打开虚拟机>设置>usb控制器-USB兼容性
把兼容性改成3.1，就可以了

6. windows的Camera依然没有focus auto
但是感觉已经很清晰了，不知道是不是错觉
