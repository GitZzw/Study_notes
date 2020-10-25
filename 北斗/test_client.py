import socket
# def msg_cb(msg):
#     receive_msg.data = msg


def clint():

    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 连接服务端
    conn.connect(('127.0.0.1', 8000))
    count = 0
    print(len('hello'))
    conn.send(b'hello')
    # 响应 | 接受服务端返回到数据
    receive_msg = conn.recv(30).decode()
    print (receive_msg)
# 关闭 socket
    conn.close()
if __name__ == "__main__":
    clint()
