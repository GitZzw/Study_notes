import socket

server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(('127.0.0.1',8000))
server.listen(5)

while True:
    print("waiting msg ...")
    conn, address = server.accept()


    # TCP/IP part1
    print(address)
    print("receive msg ...")
    recvdata = conn.recv(5).decode('utf-8')
    print(len(recvdata))


    # TCP/IP part2
    if len(recvdata)==5:
        print("now send msg ......")
        send_data = [1000,2000,3000]
        print(send_data)
        send_data_byte = bytes(0)
        for i in range(len(send_data)):
            print(send_data[i])
            senddata1000 = str(send_data[i])+','
            print(senddata1000.encode('utf-8'))
            send_data_byte += senddata1000.encode('utf-8')

        print(send_data_byte)
        conn.send(send_data_byte)
